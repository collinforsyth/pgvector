#include "postgres.h"

#include "access/relscan.h"
#include "hnsw.h"
#include "pgstat.h"
#include "storage/bufmgr.h"
#include "storage/lmgr.h"
#include "utils/float.h"
#include "utils/memutils.h"
#include "utils/guc.h"

/* Forward declarations for ACORN functions */
static List *GetScanItemsACORN(IndexScanDesc scan, Datum value);
static List *HnswSearchLayerACORN(char *base, HnswQuery *q, List *ep, int ef, int lc, 
								 Relation index, HnswSupport *support, int m, IndexScanDesc scan);
static void HnswExpandNeighborsACORN(IndexScanDesc scan, HnswNeighborArray *neighbors, visited_hash *v,
									int expansion_target, List **candidates, int *valid_candidates, List **w,
									int ef, Relation index, HnswSupport *support, char *base, HnswQuery *q, int64 *tuples);

/*
 * Algorithm 5 from paper
 */
static List *
GetScanItems(IndexScanDesc scan, Datum value)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	Relation	index = scan->indexRelation;
	HnswSupport *support = &so->support;
	List	   *ep;
	List	   *w;
	int			m;
	HnswElement entryPoint;
	char	   *base = NULL;
	HnswQuery  *q = &so->q;

	/* Get m and entry point */
	HnswGetMetaPageInfo(index, &m, &entryPoint);

	q->value = value;
	so->m = m;

	if (entryPoint == NULL)
		return NIL;

	ep = list_make1(HnswEntryCandidate(base, entryPoint, q, index, support, false));

	for (int lc = entryPoint->level; lc >= 1; lc--)
	{
		w = HnswSearchLayer(base, q, ep, 1, lc, index, support, m, false, NULL, NULL, NULL, true, NULL);
		ep = w;
	}

	return HnswSearchLayer(base, q, ep, hnsw_ef_search, 0, index, support, m, false, NULL, &so->v, hnsw_iterative_scan != HNSW_ITERATIVE_SCAN_OFF ? &so->discarded : NULL, true, &so->tuples);
}

/*
 * Resume scan at ground level with discarded candidates
 */
static List *
ResumeScanItems(IndexScanDesc scan)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	Relation	index = scan->indexRelation;
	List	   *ep = NIL;
	char	   *base = NULL;
	int			batch_size = hnsw_ef_search;

	if (pairingheap_is_empty(so->discarded))
		return NIL;

	/* Get next batch of candidates */
	for (int i = 0; i < batch_size; i++)
	{
		HnswSearchCandidate *sc;

		if (pairingheap_is_empty(so->discarded))
			break;

		sc = HnswGetSearchCandidate(w_node, pairingheap_remove_first(so->discarded));

		ep = lappend(ep, sc);
	}

	return HnswSearchLayer(base, &so->q, ep, batch_size, 0, index, &so->support, so->m, false, NULL, &so->v, &so->discarded, false, &so->tuples);
}

/*
 * Get scan value
 */
static Datum
GetScanValue(IndexScanDesc scan)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	Datum		value;

	if (scan->orderByData->sk_flags & SK_ISNULL)
		value = PointerGetDatum(NULL);
	else
	{
		value = scan->orderByData->sk_argument;

		/* Value should not be compressed or toasted */
		Assert(!VARATT_IS_COMPRESSED(DatumGetPointer(value)));
		Assert(!VARATT_IS_EXTENDED(DatumGetPointer(value)));

		/* Normalize if needed */
		if (so->support.normprocinfo != NULL)
			value = HnswNormValue(so->typeInfo, so->support.collation, value);
	}

	return value;
}

#if defined(HNSW_MEMORY)
/*
 * Show memory usage
 */
static void
ShowMemoryUsage(HnswScanOpaque so)
{
	elog(INFO, "memory: %zu KB, tuples: " INT64_FORMAT, MemoryContextMemAllocated(so->tmpCtx, false) / 1024, so->tuples);
}
#endif

/*
 * Extract predicates for included columns from scan keys
 */
void
HnswExtractPredicates(IndexScanDesc scan, HnswScanOpaque so)
{
	int			i;
	int			n_predicates = 0;
	ScanKey		scankey = scan->keyData;
	int			nscankeys = scan->numberOfKeys;
	TupleDesc	tupdesc = RelationGetDescr(scan->indexRelation);

	/* Initialize predicate fields */
	so->n_predicates = 0;
	so->predicate_keys = NULL;
	so->predicate_attr_nums = NULL;
	so->has_predicates = false;

	/* Count predicates on included columns (skip vector column at index 1) */
	for (i = 0; i < nscankeys; i++)
	{
		int			attnum = scankey[i].sk_attno;

		/* Skip vector column (attnum 1) and invalid attributes */
		if (attnum > 1 && attnum <= tupdesc->natts)
		{
			n_predicates++;
		}
	}

	/* No predicates found */
	if (n_predicates == 0)
		return;

	/* Allocate arrays for predicate information */
	so->predicate_keys = (ScanKey) palloc(n_predicates * sizeof(ScanKeyData));
	so->predicate_attr_nums = (int *) palloc(n_predicates * sizeof(int));
	so->n_predicates = n_predicates;
	so->has_predicates = true;

	/* Copy predicate scan keys and attribute numbers */
	n_predicates = 0;
	for (i = 0; i < nscankeys; i++)
	{
		int			attnum = scankey[i].sk_attno;

		/* Skip vector column (attnum 1) and invalid attributes */
		if (attnum > 1 && attnum <= tupdesc->natts)
		{
			so->predicate_keys[n_predicates] = scankey[i];
			so->predicate_attr_nums[n_predicates] = attnum;
			n_predicates++;
		}
	}
}

/*
 * Configure ACORN vs post-filtering based on predicates
 */
static void
HnswConfigureACORN(IndexScanDesc scan, HnswScanOpaque so)
{
	Relation index = scan->indexRelation;
	HnswOptions *opts = (HnswOptions *) index->rd_options;
	
	/* Set ACORN parameters from index options or GUCs */
	so->gamma = opts ? opts->gamma : hnsw_gamma;
	
	/* Simple decision: use ACORN if we have predicates */
	if (so->has_predicates)
	{
		so->use_acorn = true;
		elog(DEBUG2, "HNSW: Using ACORN-1 with gamma=%d for %d predicates", 
			 so->gamma, so->n_predicates);
	}
	else
	{
		so->use_acorn = false;
		elog(DEBUG2, "HNSW: No predicates, using standard HNSW search");
	}
	
	/* Reset ACORN tracking counters */
	so->predicate_passes = 0;
	so->predicate_failures = 0;
	so->expansions_performed = 0;
}

/*
 * Prepare for an index scan
 */
IndexScanDesc
hnswbeginscan(Relation index, int nkeys, int norderbys)
{
	IndexScanDesc scan;
	HnswScanOpaque so;
	double		maxMemory;

	scan = RelationGetIndexScan(index, nkeys, norderbys);

	so = (HnswScanOpaque) palloc(sizeof(HnswScanOpaqueData));
	so->typeInfo = HnswGetTypeInfo(index);

	/* Set support functions */
	HnswInitSupport(&so->support, index);

	/* Initialize predicate fields */
	so->n_predicates = 0;
	so->predicate_keys = NULL;
	so->predicate_attr_nums = NULL;
	so->has_predicates = false;
	so->index_tuple_desc = NULL;

	/* Initialize performance tracking fields */
	so->candidates_examined = 0;
	so->candidates_filtered = 0;
	so->results_returned = 0;
	so->search_expanded = false;

	/*
	 * Use a lower max allocation size than default to allow scanning more
	 * tuples for iterative search before exceeding work_mem
	 */
	so->tmpCtx = AllocSetContextCreate(CurrentMemoryContext,
									   "Hnsw scan temporary context",
									   0, 8 * 1024, 256 * 1024);

	/* Calculate max memory */
	/* Add 256 extra bytes to fill last block when close */
	maxMemory = (double) work_mem * hnsw_scan_mem_multiplier * 1024.0 + 256;
	so->maxMemory = Min(maxMemory, (double) SIZE_MAX);

	scan->opaque = so;

	return scan;
}

/*
 * Start or restart an index scan
 */
void
hnswrescan(IndexScanDesc scan, ScanKey keys, int nkeys, ScanKey orderbys, int norderbys)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;

	so->first = true;
	/* v and discarded are allocated in tmpCtx */
	so->v.tids = NULL;
	so->discarded = NULL;
	so->tuples = 0;
	so->previousDistance = -get_float8_infinity();
	MemoryContextReset(so->tmpCtx);

	/* Reset performance tracking */
	so->candidates_examined = 0;
	so->candidates_filtered = 0;
	so->results_returned = 0;
	so->search_expanded = false;

	if (keys && scan->numberOfKeys > 0)
		memmove(scan->keyData, keys, scan->numberOfKeys * sizeof(ScanKeyData));

	if (orderbys && scan->numberOfOrderBys > 0)
		memmove(scan->orderByData, orderbys, scan->numberOfOrderBys * sizeof(ScanKeyData));

	/* Extract predicates for included columns */
	HnswExtractPredicates(scan, so);

	/* Initialize tuple descriptor for included columns if we have predicates */
	if (so->has_predicates && so->index_tuple_desc == NULL)
	{
		so->index_tuple_desc = RelationGetDescr(scan->indexRelation);
	}

	/* Configure ACORN vs post-filtering based on predicates */
	HnswConfigureACORN(scan, so);
}

/*
 * Expand search for highly selective predicates
 */
static List *
HnswExpandSearch(IndexScanDesc scan)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	Datum		value;
	int			original_ef = hnsw_ef_search;
	List	   *expanded_w;

	/* Expand ef parameter for more candidates */
	hnsw_ef_search = Min(hnsw_ef_search * 2, 1000);	/* Cap at reasonable limit */

	elog(DEBUG2, "HNSW: Expanding search from ef=%d to ef=%d due to high predicate selectivity",
		 original_ef, hnsw_ef_search);

	/* Get scan value */
	value = GetScanValue(scan);

	/*
	 * Get a shared lock. This allows vacuum to ensure no in-flight scans
	 * before marking tuples as deleted.
	 */
	LockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);

	/* Perform expanded search */
	expanded_w = GetScanItems(scan, value);

	/* Release shared lock */
	UnlockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);

	/* Restore original ef value */
	hnsw_ef_search = original_ef;

	/* Mark that search was expanded */
	so->search_expanded = true;

	return expanded_w;
}

/*
 * Fetch the next tuple in the given scan
 */
bool
hnswgettuple(IndexScanDesc scan, ScanDirection dir)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	MemoryContext oldCtx = MemoryContextSwitchTo(so->tmpCtx);

	/*
	 * Index can be used to scan backward, but Postgres doesn't support
	 * backward scan on operators
	 */
	Assert(ScanDirectionIsForward(dir));

	if (so->first)
	{
		Datum		value;

		/* Count index scan for stats */
		pgstat_count_index_scan(scan->indexRelation);
#if PG_VERSION_NUM >= 180000
		if (scan->instrument)
			scan->instrument->nsearches++;
#endif

		/* Safety check */
		if (scan->orderByData == NULL)
			elog(ERROR, "cannot scan hnsw index without order");

		/* Requires MVCC-compliant snapshot as not able to maintain a pin */
		/* https://www.postgresql.org/docs/current/index-locking.html */
		if (!IsMVCCSnapshot(scan->xs_snapshot))
			elog(ERROR, "non-MVCC snapshots are not supported with hnsw");

		/* Get scan value */
		value = GetScanValue(scan);

		/*
		 * Get a shared lock. This allows vacuum to ensure no in-flight scans
		 * before marking tuples as deleted.
		 */
		LockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);

		/* Choose search algorithm based on ACORN configuration */
		if (so->use_acorn)
		{
			elog(DEBUG2, "HNSW: Starting ACORN-1 search with gamma=%d", so->gamma);
			/* For now, use standard search but log that we're in ACORN mode */
			so->w = GetScanItemsACORN(scan, value);
		}
		else
		{
			elog(DEBUG2, "HNSW: Starting standard HNSW search");
		so->w = GetScanItems(scan, value);
		}

		/* Release shared lock */
		UnlockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);

		so->first = false;

#if defined(HNSW_MEMORY)
		ShowMemoryUsage(so);
#endif
	}

	for (;;)
	{
		char	   *base = NULL;
		HnswSearchCandidate *sc;
		HnswElement element;
		ItemPointer heaptid;

		if (list_length(so->w) == 0)
		{
			if (hnsw_iterative_scan == HNSW_ITERATIVE_SCAN_OFF)
			{
				/* Check if we should expand search for highly selective predicates */
				if (so->has_predicates && !so->search_expanded && 
					so->candidates_examined < hnsw_ef_search * 2 && 
					so->candidates_examined > 0)
				{
					elog(DEBUG2, "HNSW: All %d candidates filtered out by predicates, expanding search", so->candidates_examined);
					so->w = HnswExpandSearch(scan);
					if (list_length(so->w) > 0)
						continue;
				}
				break;
			}

			/* Empty index */
			if (so->discarded == NULL)
				break;

			/* Reached max number of tuples or memory limit */
			if (so->tuples >= hnsw_max_scan_tuples || MemoryContextMemAllocated(so->tmpCtx, false) > so->maxMemory)
			{
				if (pairingheap_is_empty(so->discarded))
					break;

				/* Return remaining tuples */
				so->w = lappend(so->w, HnswGetSearchCandidate(w_node, pairingheap_remove_first(so->discarded)));
			}
			else
			{
				/*
				 * Locking ensures when neighbors are read, the elements they
				 * reference will not be deleted (and replaced) during the
				 * iteration.
				 *
				 * Elements loaded into memory on previous iterations may have
				 * been deleted (and replaced), so when reading neighbors, the
				 * element version must be checked.
				 */
				LockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);

				so->w = ResumeScanItems(scan);

				UnlockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);

#if defined(HNSW_MEMORY)
				ShowMemoryUsage(so);
#endif
			}

			if (list_length(so->w) == 0)
				break;
		}

		sc = llast(so->w);
		element = HnswPtrAccess(base, sc->element);

		/* Move to next element if no valid heap TIDs */
		if (element->heaptidsLength == 0)
		{
			so->w = list_delete_last(so->w);

			/* Mark memory as free for next iteration */
			if (hnsw_iterative_scan != HNSW_ITERATIVE_SCAN_OFF)
			{
				pfree(element);
				pfree(sc);
			}

			continue;
		}

		heaptid = &element->heaptids[--element->heaptidsLength];

		/* ACORN-1 vs Post-filtering predicate evaluation */
		if (so->has_predicates)
		{
			ItemPointerData indextid;
			IndexTuple	itup;
			
			so->candidates_examined++;
			
			/* Create ItemPointer for the index tuple */
			ItemPointerSet(&indextid, element->blkno, element->offno);
			
			/* Load the IndexTuple for predicate evaluation */
			itup = HnswGetIndexTuple(scan->indexRelation, &indextid);
			
			/* If no IndexTuple available, skip this candidate */
			if (itup == NULL)
			{
				so->candidates_filtered++;
				continue;
			}
			
			/* Evaluate predicates on the IndexTuple */
			if (!HnswEvaluatePredicates(so, itup))
			{
				so->candidates_filtered++;
				/* For ACORN mode, count this as a predicate failure */
				if (so->use_acorn)
					so->predicate_failures++;
				/* IndexTuple cleanup handled by memory context */
				continue;
			}
			
			/* Predicate passed */
			so->results_returned++;
			if (so->use_acorn)
				so->predicate_passes++;
			
			/* Set up IndexTuple for INCLUDE column access if needed */
			scan->xs_itup = itup;
			scan->xs_itupdesc = so->index_tuple_desc;
		}

		/* Check if caller wants index tuple data for index-only scan */
		if (scan->xs_want_itup && !so->has_predicates)
		{
			TupleDesc	tupdesc = RelationGetDescr(scan->indexRelation);
			
			/* Only try to get IndexTuple if the index has INCLUDE columns */
			if (tupdesc->natts > 1)
			{
				ItemPointerData indextid;
				IndexTuple	itup;
				
				/* Create ItemPointer for the index tuple */
				ItemPointerSet(&indextid, element->blkno, element->offno);
				
				/* Load the IndexTuple for index-only scan */
				itup = HnswGetIndexTuple(scan->indexRelation, &indextid);
				
				/* Set up IndexTuple for index-only scan */
				if (itup != NULL)
				{
					scan->xs_itup = itup;
					if (so->index_tuple_desc == NULL)
						so->index_tuple_desc = tupdesc;
					scan->xs_itupdesc = so->index_tuple_desc;
				}
			}
		}

		if (hnsw_iterative_scan == HNSW_ITERATIVE_SCAN_STRICT)
		{
			if (sc->distance < so->previousDistance)
				continue;

			so->previousDistance = sc->distance;
		}

		MemoryContextSwitchTo(oldCtx);

		scan->xs_heaptid = *heaptid;
		scan->xs_recheck = false;
		scan->xs_recheckorderby = false;
		return true;
	}

	MemoryContextSwitchTo(oldCtx);
	
	/* Enhanced performance logging for ACORN vs post-filtering */
	if (so->has_predicates && log_min_messages <= DEBUG1)
	{
		double		selectivity = so->candidates_examined > 0 ?
			(double) so->results_returned / so->candidates_examined : 0.0;

		if (so->use_acorn)
		{
			elog(DEBUG1, "ACORN-1 stats: examined=%d, returned=%d, predicate_passes=%d, predicate_failures=%d, "
						"expansions=%d, selectivity=%.3f, gamma=%d",
				 so->candidates_examined, so->results_returned, so->predicate_passes, 
				 so->predicate_failures, so->expansions_performed, selectivity, so->gamma);
		}
		else
		{
		elog(DEBUG1, "HNSW post-filtering stats: examined=%d, filtered=%d, returned=%d, selectivity=%.3f, expanded=%s",
			 so->candidates_examined,
			 so->candidates_filtered,
			 so->results_returned,
			 selectivity,
			 so->search_expanded ? "yes" : "no");
		}
	}
	
	return false;
}

/*
 * End a scan and release resources
 */
void
hnswendscan(IndexScanDesc scan)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;

	/* Final performance logging if we have predicates */
	if (so->has_predicates && so->candidates_examined > 0 && log_min_messages <= DEBUG2)
	{
		double		selectivity = so->candidates_examined > 0 ?
			(double) so->results_returned / so->candidates_examined : 0.0;

		elog(DEBUG2, "HNSW scan final stats: examined=%d, filtered=%d, returned=%d, selectivity=%.3f",
			 so->candidates_examined,
			 so->candidates_filtered,
			 so->results_returned,
			 selectivity);
	}

	MemoryContextDelete(so->tmpCtx);

	pfree(so);
	scan->opaque = NULL;
}

/*
 * ACORN-1 Search Items with predicate evaluation during traversal
 * This is a simplified implementation that uses predicate evaluation during search
 */
static List *
GetScanItemsACORN(IndexScanDesc scan, Datum value)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	Relation	index = scan->indexRelation;
	HnswSupport *support = &so->support;
	List	   *ep;
	List	   *w;
	int			m;
	HnswElement entryPoint;
	char	   *base = NULL;
	HnswQuery  *q = &so->q;

	/* Get m and entry point */
	HnswGetMetaPageInfo(index, &m, &entryPoint);

	q->value = value;
	so->m = m;

	if (entryPoint == NULL)
		return NIL;

	ep = list_make1(HnswEntryCandidate(base, entryPoint, q, index, support, false));

	/* Search upper layers (same as standard HNSW) */
	for (int lc = entryPoint->level; lc >= 1; lc--)
	{
		w = HnswSearchLayer(base, q, ep, 1, lc, index, support, m, false, NULL, NULL, NULL, true, NULL);
		ep = w;
	}

	/* ACORN-1: Use modified search at layer 0 with predicate-aware expansion */
	return HnswSearchLayerACORN(base, q, ep, hnsw_ef_search, 0, index, support, m, scan);
}

/*
 * ACORN-1 search layer with predicate evaluation during traversal
 * This modifies the standard search to expand neighbors when predicates filter connections
 */
static List *
HnswSearchLayerACORN(char *base, HnswQuery *q, List *ep, int ef, int lc, 
					 Relation index, HnswSupport *support, int m, IndexScanDesc scan)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	List	   *w = NIL;
	List	   *candidates = NIL;
	ListCell   *lc2;
	visited_hash *v = &so->v;
	int64		tuples = 0;
	int			valid_candidates = 0;

	/* Use existing visited hash if available */
	if (v->tids == NULL)
	{
		v->tids = tidhash_create(CurrentMemoryContext, ef + m, NULL);
	}

	/* Process entry points with ACORN-1 predicate evaluation */
	foreach(lc2, ep)
	{
		HnswSearchCandidate *sc = (HnswSearchCandidate *) lfirst(lc2);
		HnswElement element = HnswPtrAccess(base, sc->element);
		ItemPointerData tid;
		TidHashEntry *entry;
		bool		found;

		ItemPointerSet(&tid, element->blkno, element->offno);

		/* Check if already visited */
		entry = tidhash_insert(v->tids, tid, &found);
		if (found)
			continue;

		entry->status = 1;
		tuples++;

		/* ACORN-1: Evaluate predicates on entry point */
		if (so->has_predicates)
		{
			IndexTuple itup = HnswGetIndexTuple(index, &tid);
			if (itup != NULL && HnswEvaluatePredicates(so, itup))
			{
				so->predicate_passes++;
				candidates = lappend(candidates, sc);
				valid_candidates++;
			}
			else
			{
				so->predicate_failures++;
				/* Skip this candidate but continue search */
			}
		}
		else
		{
			candidates = lappend(candidates, sc);
			valid_candidates++;
		}

		w = lappend(w, sc);
	}

	/* ACORN-1 main search loop with neighbor expansion */
	while (list_length(w) > 0)
	{
		HnswSearchCandidate *sc;
		HnswElement element;
		HnswNeighborArray *neighbors;
		int			neighbor_idx;
		int			valid_neighbors_found = 0;

		/* Get next candidate to explore */
		sc = (HnswSearchCandidate *) linitial(w);
		w = list_delete_first(w);
		element = HnswPtrAccess(base, sc->element);

		/* Stop if we have enough valid candidates and this one is far */
		if (valid_candidates >= ef && list_length(candidates) > 0)
		{
			HnswSearchCandidate *farthest = (HnswSearchCandidate *) llast(candidates);
			if (sc->distance > farthest->distance)
				break;
		}

		/* Get neighbors for this element */
		neighbors = HnswGetNeighbors(base, element, lc);

		/* ACORN-1: Process neighbors with predicate evaluation */
		for (neighbor_idx = 0; neighbor_idx < neighbors->length && valid_neighbors_found < so->gamma; neighbor_idx++)
		{
			HnswCandidate *neighbor_candidate = &neighbors->items[neighbor_idx];
			HnswElement neighbor = HnswPtrAccess(base, neighbor_candidate->element);
			ItemPointerData ntid;
			TidHashEntry *entry;
			bool		found;
			double		distance;

			ItemPointerSet(&ntid, neighbor->blkno, neighbor->offno);

			/* Check if already visited */
			entry = tidhash_insert(v->tids, ntid, &found);
			if (found)
				continue;

			entry->status = 1;
			tuples++;

			/* ACORN-1: Evaluate predicates on neighbor */
			if (so->has_predicates)
			{
				IndexTuple itup = HnswGetIndexTuple(index, &ntid);
				if (itup == NULL || !HnswEvaluatePredicates(so, itup))
				{
					so->predicate_failures++;
					continue; /* Skip this neighbor */
				}
				so->predicate_passes++;
			}

			/* Valid neighbor - calculate distance and add to search */
			valid_neighbors_found++;
			HnswLoadElement(neighbor, &distance, q, index, support, true, NULL);

			/* Create candidate for this valid neighbor */
			HnswSearchCandidate *nsc = HnswEntryCandidate(base, neighbor, q, index, support, false);
			nsc->distance = distance;

			/* Add to candidates list (maintain sorted order) */
			if (valid_candidates < ef)
			{
				candidates = lappend(candidates, nsc);
				valid_candidates++;
			}
			else
			{
				/* Replace farthest candidate if this one is closer */
				HnswSearchCandidate *farthest = (HnswSearchCandidate *) llast(candidates);
				if (nsc->distance < farthest->distance)
				{
					candidates = list_delete_last(candidates);
					candidates = lappend(candidates, nsc);
				}
			}

			/* Add to exploration queue */
			w = lappend(w, nsc);
		}

		/* ACORN-1: Expand neighbors if we didn't find enough valid ones */
		if (so->has_predicates && valid_neighbors_found < so->gamma && neighbors->length > 0)
		{
			so->expansions_performed++;
			elog(DEBUG3, "ACORN: Expanding search at node (found %d, target %d)", 
				 valid_neighbors_found, so->gamma);
			
			/* Simple expansion: examine neighbors of neighbors up to gamma limit */
			HnswExpandNeighborsACORN(scan, neighbors, v, so->gamma - valid_neighbors_found, 
									&candidates, &valid_candidates, &w, ef, index, support, base, q, &tuples);
		}
	}

	so->tuples = tuples;
	return candidates;
}

/*
 * ACORN-1 neighbor expansion: examine second-degree neighbors
 */
static void
HnswExpandNeighborsACORN(IndexScanDesc scan, HnswNeighborArray *neighbors, visited_hash *v,
						int expansion_target, List **candidates, int *valid_candidates, List **w,
						int ef, Relation index, HnswSupport *support, char *base, HnswQuery *q, int64 *tuples)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	int			found_count = 0;
	int			neighbor_idx;

	/* Examine neighbors of neighbors */
	for (neighbor_idx = 0; neighbor_idx < neighbors->length && found_count < expansion_target; neighbor_idx++)
	{
		HnswCandidate *neighbor_candidate = &neighbors->items[neighbor_idx];
		HnswElement neighbor = HnswPtrAccess(base, neighbor_candidate->element);
		HnswNeighborArray *second_neighbors;
		int			second_idx;

		/* Get second-degree neighbors */
		second_neighbors = HnswGetNeighbors(base, neighbor, 0);

		/* Process second-degree neighbors */
		for (second_idx = 0; second_idx < second_neighbors->length && found_count < expansion_target; second_idx++)
		{
			HnswCandidate *second_candidate = &second_neighbors->items[second_idx];
			HnswElement second_neighbor = HnswPtrAccess(base, second_candidate->element);
			ItemPointerData tid;
			TidHashEntry *entry;
			bool		found;
			double		distance;

			ItemPointerSet(&tid, second_neighbor->blkno, second_neighbor->offno);

			/* Check if already visited */
			entry = tidhash_insert(v->tids, tid, &found);
			if (found)
				continue;

			entry->status = 1;
			(*tuples)++;

			/* Evaluate predicates on second-degree neighbor */
			if (so->has_predicates)
			{
				IndexTuple itup = HnswGetIndexTuple(index, &tid);
				if (itup == NULL || !HnswEvaluatePredicates(so, itup))
				{
					so->predicate_failures++;
					continue;
				}
				so->predicate_passes++;
			}

			/* Found valid expanded neighbor */
			found_count++;
			HnswLoadElement(second_neighbor, &distance, q, index, support, true, NULL);

			/* Create candidate for this expanded neighbor */
			HnswSearchCandidate *sc = HnswEntryCandidate(base, second_neighbor, q, index, support, false);
			sc->distance = distance;

			/* Add to candidates if we have room or it's better than worst */
			if (*valid_candidates < ef)
			{
				*candidates = lappend(*candidates, sc);
				(*valid_candidates)++;
			}
			else
			{
				HnswSearchCandidate *farthest = (HnswSearchCandidate *) llast(*candidates);
				if (sc->distance < farthest->distance)
				{
					*candidates = list_delete_last(*candidates);
					*candidates = lappend(*candidates, sc);
				}
			}

			/* Add to exploration queue */
			*w = lappend(*w, sc);

			elog(DEBUG4, "ACORN: Found expanded neighbor at distance %.3f", sc->distance);
		}
	}

	elog(DEBUG3, "ACORN: Expansion found %d additional neighbors (target was %d)", 
		 found_count, expansion_target);
}
