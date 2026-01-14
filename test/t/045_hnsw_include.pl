use strict;
use warnings FATAL => 'all';
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

# Initialize node
my $node = PostgreSQL::Test::Cluster->new('node');
$node->init;
$node->start;

# Create extension and test table
$node->safe_psql("postgres", "CREATE EXTENSION vector;");
$node->safe_psql("postgres", "CREATE TABLE test_items (id int, embedding vector(3), category text);");

# Insert simple test data
$node->safe_psql("postgres", "INSERT INTO test_items VALUES (1, '[1,0,0]', 'books'), (2, '[0,1,0]', 'electronics');");

# Test 1: Create HNSW index with INCLUDE columns - should not error
$node->safe_psql("postgres", "CREATE INDEX test_hnsw_idx ON test_items USING hnsw (embedding vector_l2_ops) INCLUDE (category);");
pass("HNSW index with INCLUDE column created without errors");

# Test 2: Verify the index was created and shows up in system catalogs
my $index_exists = $node->safe_psql("postgres", "SELECT indexname FROM pg_indexes WHERE indexname = 'test_hnsw_idx';");
is($index_exists, "test_hnsw_idx", "HNSW index with INCLUDE column exists in pg_indexes");

# Test 3: Verify the index definition contains INCLUDE
my $index_def = $node->safe_psql("postgres", "SELECT indexdef FROM pg_indexes WHERE indexname = 'test_hnsw_idx';");
like($index_def, qr/INCLUDE \(category\)/, "Index definition shows INCLUDE column");

# Test 4: Basic vector query works (even if not optimally)
my $query_result = $node->safe_psql("postgres", "SELECT id FROM test_items ORDER BY embedding <-> '[1,0,0]' LIMIT 1;");
is($query_result, "1", "Basic vector similarity query works");

$node->stop;
done_testing();
