#include "postgres.h"

#include "access/relscan.h"
#include "hnsw.h"
#include "pgstat.h"
#include "storage/bufmgr.h"
#include "storage/lmgr.h"
#include "utils/float.h"
#include "utils/memutils.h"
#include "utils/guc.h"

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
	/* TEMPORARY: Disable predicate evaluation to isolate the crash issue */
	/* Initialize predicate fields */
	so->n_predicates = 0;
	so->predicate_keys = NULL;
	so->predicate_attr_nums = NULL;
	so->has_predicates = false;
	return;
	
	/* Original code disabled for now */
#if 0
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
#endif
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

		so->w = GetScanItems(scan, value);

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

		/* NEW: Post-filtering with predicate evaluation */
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
				/* IndexTuple cleanup handled by memory context */
				continue;
			}
			
			/* Predicate passed */
			so->results_returned++;
			
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
	
	/* Performance logging for post-filtering */
	if (so->has_predicates && log_min_messages <= DEBUG1)
	{
		double		selectivity = so->candidates_examined > 0 ?
			(double) so->results_returned / so->candidates_examined : 0.0;

		elog(DEBUG1, "HNSW post-filtering stats: examined=%d, filtered=%d, returned=%d, selectivity=%.3f, expanded=%s",
			 so->candidates_examined,
			 so->candidates_filtered,
			 so->results_returned,
			 selectivity,
			 so->search_expanded ? "yes" : "no");
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
