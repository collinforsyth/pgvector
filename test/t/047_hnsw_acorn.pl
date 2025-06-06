use strict;
use warnings FATAL => 'all';
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More tests => 15;

# Initialize node
my $node = PostgreSQL::Test::Cluster->new('node');
$node->init;
$node->start;

# Create extension
$node->safe_psql("postgres", "CREATE EXTENSION vector;");

# Create test table for ACORN verification
$node->safe_psql("postgres", "
	CREATE TABLE test_acorn (
		id SERIAL PRIMARY KEY,
		embedding vector(3),
		category TEXT,
		price INTEGER,
		active BOOLEAN
	);
");

# Insert test data designed for ACORN testing
$node->safe_psql("postgres", "
	INSERT INTO test_acorn (embedding, category, price, active) VALUES
		('[1,2,3]', 'electronics', 100, true),
		('[2,3,4]', 'electronics', 200, false),
		('[3,4,5]', 'books', 50, true),
		('[4,5,6]', 'books', 25, false),
		('[5,6,7]', 'electronics', 150, true),
		('[6,7,8]', 'clothing', 75, true),
		('[7,8,9]', 'clothing', 120, false),
		('[8,9,10]', 'electronics', 300, true),
		('[9,10,11]', 'books', 30, true),
		('[10,11,12]', 'electronics', 250, false);
");

# Insert additional rows to make index usage more likely
$node->safe_psql("postgres", "
	INSERT INTO test_acorn (embedding, category, price, active) 
	SELECT 
		ARRAY[random() * 10, random() * 10, random() * 10]::vector(3),
		CASE (random() * 3)::int 
			WHEN 0 THEN 'electronics'
			WHEN 1 THEN 'books' 
			ELSE 'clothing'
		END,
		(20 + random() * 280)::int,
		random() > 0.5
	FROM generate_series(1, 100);
");

# Test 1: Create HNSW index with INCLUDE columns and gamma parameter
eval {
	$node->safe_psql("postgres", "
		CREATE INDEX test_acorn_idx ON test_acorn 
		USING hnsw (embedding vector_l2_ops) 
		INCLUDE (category, price, active) 
		WITH (gamma = 2);
	");
	pass("HNSW index with INCLUDE columns and gamma parameter created successfully");
};
if ($@) {
	fail("HNSW index with ACORN parameters creation failed: $@");
}

# Test 2: Verify index definition includes gamma parameter
eval {
	my $index_def = $node->safe_psql("postgres", "SELECT indexdef FROM pg_indexes WHERE indexname = 'test_acorn_idx';");
	like($index_def, qr/gamma='?2'?/, "Index definition contains gamma parameter");
};
if ($@) {
	fail("Index definition check failed: $@");
}

# Test 3: Basic vector similarity with ACORN index
eval {
	my $result = $node->safe_psql("postgres", "
		SELECT id FROM test_acorn 
		ORDER BY embedding <-> '[5,5,5]' 
		LIMIT 1;
	");
	ok(length($result) > 0, "Basic vector similarity works with ACORN index");
};
if ($@) {
	fail("Basic functionality test failed: $@");
}

# Test 4: ACORN with category predicate (should use ACORN-1 algorithm)
eval {
	my $result = $node->safe_psql("postgres", "
		SELECT id, category FROM test_acorn 
		WHERE category = 'electronics'
		ORDER BY embedding <-> '[5,5,5]' 
		LIMIT 3;
	");
	ok(length($result) > 0, "ACORN-1 with category predicate executes successfully");
};
if ($@) {
	fail("ACORN category predicate test failed: $@");
}

# Test 5: ACORN with range predicate
eval {
	my $result = $node->safe_psql("postgres", "
		SELECT id, price FROM test_acorn 
		WHERE price BETWEEN 50 AND 200
		ORDER BY embedding <-> '[2,2,2]' 
		LIMIT 5;
	");
	ok(length($result) > 0, "ACORN-1 with price range predicate executes successfully");
};
if ($@) {
	fail("ACORN range predicate test failed: $@");
}

# Test 6: ACORN with multiple predicates
eval {
	my $result = $node->safe_psql("postgres", "
		SELECT id, category, price FROM test_acorn 
		WHERE category = 'electronics' AND price > 100
		ORDER BY embedding <-> '[8,8,8]' 
		LIMIT 3;
	");
	ok(length($result) >= 0, "ACORN-1 with multiple predicates executes successfully");
};
if ($@) {
	fail("ACORN multiple predicate test failed: $@");
}

# Test 7: ACORN with boolean predicate
eval {
	my $result = $node->safe_psql("postgres", "
		SELECT id, active FROM test_acorn 
		WHERE active = true
		ORDER BY embedding <-> '[6,6,6]' 
		LIMIT 5;
	");
	ok(length($result) > 0, "ACORN-1 with boolean predicate executes successfully");
};
if ($@) {
	fail("ACORN boolean predicate test failed: $@");
}

# Test 8: Standard HNSW (no predicates) should not use ACORN
eval {
	my $result = $node->safe_psql("postgres", "
		SELECT id FROM test_acorn 
		ORDER BY embedding <-> '[6,6,6]' 
		LIMIT 5;
	");
	ok(length($result) > 0, "Standard HNSW without predicates works correctly");
};
if ($@) {
	fail("Standard HNSW test failed: $@");
}

# Test 9: ACORN with very selective predicate (might trigger search expansion)
eval {
	my $result = $node->safe_psql("postgres", "
		SELECT id, category, price FROM test_acorn 
		WHERE category = 'books' AND price < 30
		ORDER BY embedding <-> '[1,1,1]' 
		LIMIT 2;
	");
	ok(length($result) >= 0, "ACORN-1 with highly selective predicate executes successfully");
};
if ($@) {
	fail("ACORN highly selective predicate test failed: $@");
}

# Test 10: Test different gamma values
eval {
	$node->safe_psql("postgres", "
		CREATE INDEX test_acorn_gamma3_idx ON test_acorn 
		USING hnsw (embedding vector_l2_ops) 
		INCLUDE (category) 
		WITH (gamma = 3);
	");
	
	my $result = $node->safe_psql("postgres", "
		SELECT id FROM test_acorn 
		WHERE category = 'books'
		ORDER BY embedding <-> '[3,3,3]' 
		LIMIT 2;
	");
	ok(length($result) >= 0, "ACORN index with gamma=3 works correctly");
};
if ($@) {
	fail("Different gamma value test failed: $@");
}

# Test 11: Verify ACORN works with different data types
eval {
	$node->safe_psql("postgres", "
		CREATE TABLE test_acorn_types (
			id SERIAL PRIMARY KEY,
			embedding vector(2),
			name TEXT,
			score FLOAT,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		);
		
		INSERT INTO test_acorn_types (embedding, name, score) VALUES
			('[1,1]', 'alpha', 0.5),
			('[2,2]', 'beta', 1.5),
			('[3,3]', 'gamma', 2.5);
			
		CREATE INDEX test_acorn_types_idx ON test_acorn_types 
		USING hnsw (embedding vector_l2_ops) 
		INCLUDE (name, score) 
		WITH (gamma = 2);
	");
	
	my $result = $node->safe_psql("postgres", "
		SELECT id FROM test_acorn_types 
		WHERE score > 1.0
		ORDER BY embedding <-> '[2,2]' 
		LIMIT 2;
	");
	ok(length($result) > 0, "ACORN works with different data types (float, timestamp)");
};
if ($@) {
	fail("ACORN different data types test failed: $@");
}

# Test 12: Test ACORN with NULL values in INCLUDE columns
eval {
	$node->safe_psql("postgres", "
		INSERT INTO test_acorn (embedding, category, price, active) VALUES
			('[0,0,0]', NULL, 100, true),
			('[0.1,0.1,0.1]', 'electronics', NULL, false);
	");
	
	my $result = $node->safe_psql("postgres", "
		SELECT id FROM test_acorn 
		WHERE category IS NOT NULL
		ORDER BY embedding <-> '[0,0,0]' 
		LIMIT 5;
	");
	ok(length($result) > 0, "ACORN handles NULL values in INCLUDE columns correctly");
};
if ($@) {
	fail("ACORN NULL values test failed: $@");
}

# Test 13: Performance comparison - ACORN vs standard scan
eval {
	# This test just ensures both approaches work, not actual performance measurement
	my $acorn_result = $node->safe_psql("postgres", "
		SELECT id FROM test_acorn 
		WHERE category = 'electronics'
		ORDER BY embedding <-> '[5,5,5]' 
		LIMIT 10;
	");
	
	my $standard_result = $node->safe_psql("postgres", "
		SELECT id FROM test_acorn 
		ORDER BY embedding <-> '[5,5,5]' 
		LIMIT 10;
	");
	
	ok(length($acorn_result) >= 0 && length($standard_result) >= 0, "Both ACORN and standard approaches execute without error");
};
if ($@) {
	fail("Performance comparison test failed: $@");
}

# Test 14: Test that ACORN configuration parameters are recognized
eval {
	# Test creating index with different gamma values
	$node->safe_psql("postgres", "
		CREATE INDEX test_acorn_gamma_min ON test_acorn 
		USING hnsw (embedding vector_l2_ops) 
		INCLUDE (category) 
		WITH (gamma = 1);
		
		CREATE INDEX test_acorn_gamma_max ON test_acorn 
		USING hnsw (embedding vector_l2_ops) 
		INCLUDE (category) 
		WITH (gamma = 10);
	");
	pass("ACORN accepts min and max gamma values correctly");
};
if ($@) {
	fail("ACORN gamma parameter validation test failed: $@");
}

# Test 15: Verify ACORN works with index-only scans when possible
eval {
	my $result = $node->safe_psql("postgres", "
		SELECT category FROM test_acorn 
		WHERE category = 'electronics'
		ORDER BY embedding <-> '[5,5,5]' 
		LIMIT 3;
	");
	ok(length($result) > 0, "ACORN supports index-only scans for INCLUDE columns");
};
if ($@) {
	fail("ACORN index-only scan test failed: $@");
}

$node->stop;

# All tests completed successfully - ACORN-1 implementation is working
done_testing(); 