#include <shogun/io/SGIO.h>
#include <shogun/kernel/CachedKernel.h>

using namespace shogun;

#ifdef USE_SVMLIGHT

CachedKernel::CachedKernel(CKernel* k)
{
	kernel = k;
	SG_REF(kernel);
	memset(&kernel_cache, 0x0, sizeof(KERNEL_CACHE));
}

CachedKernel::CachedKernel(CachedKernel&& orig)
    : kernel(orig.kernel), kernel_cache(orig.kernel_cache)
{
	memset(&orig.kernel_cache, 0x0, sizeof(KERNEL_CACHE));
	orig.kernel = nullptr;
}

CachedKernel::~CachedKernel()
{
	set_kernel(nullptr);
}

void CachedKernel::set_kernel(CKernel* k)
{
	SG_UNREF(kernel);
	kernel = k;
	SG_REF(kernel);
	kernel_cache_cleanup();
}

void CachedKernel::resize_kernel_cache(
    KERNELCACHE_IDX size, bool regression_hack)
{
	if (size < 10)
		size = 10;

	kernel_cache_cleanup();
	kernel->set_cache_size(size);

	if (kernel->has_features() && kernel->get_num_vec_lhs())
		kernel_cache_init(kernel->get_cache_size(), regression_hack);
}

void CachedKernel::kernel_cache_init(int32_t buffsize, bool regression_hack)
{
	int32_t totdoc = kernel->get_num_vec_lhs();
	REQUIRE(
	    totdoc > 0, "kernel has zero rows: num_lhs=%d num_rhs=%d\n",
	    kernel->get_num_vec_lhs(), kernel->get_num_vec_rhs())

	uint64_t buffer_size = 0;
	int32_t i;

	// in regression the additional constraints are made by doubling the
	// training data
	if (regression_hack)
		totdoc *= 2;

	buffer_size = ((uint64_t)buffsize) * 1024 * 1024 / sizeof(KERNELCACHE_ELEM);
	if (buffer_size > ((uint64_t)totdoc) * totdoc)
		buffer_size = ((uint64_t)totdoc) * totdoc;

	SG_SINFO(
	    "using a kernel cache of size %lld MB (%lld bytes) for %s Kernel\n",
	    buffer_size * sizeof(KERNELCACHE_ELEM) / 1024 / 1024,
	    buffer_size * sizeof(KERNELCACHE_ELEM), kernel->get_name())

	// make sure it fits in the *signed* KERNELCACHE_IDX type
	ASSERT(buffer_size < (((uint64_t)1) << (sizeof(KERNELCACHE_IDX) * 8 - 1)))

	kernel_cache.index = SG_MALLOC(int32_t, totdoc);
	kernel_cache.occu = SG_MALLOC(int32_t, totdoc);
	kernel_cache.lru = SG_MALLOC(int32_t, totdoc);
	kernel_cache.invindex = SG_MALLOC(int32_t, totdoc);
	kernel_cache.active2totdoc = SG_MALLOC(int32_t, totdoc);
	kernel_cache.totdoc2active = SG_MALLOC(int32_t, totdoc);
	kernel_cache.buffer = SG_MALLOC(KERNELCACHE_ELEM, buffer_size);
	kernel_cache.buffsize = buffer_size;
	kernel_cache.max_elems = (int32_t)(kernel_cache.buffsize / totdoc);

	if (kernel_cache.max_elems > totdoc)
	{
		kernel_cache.max_elems = totdoc;
	}

	kernel_cache.elems = 0; // initialize cache
	for (i = 0; i < totdoc; i++)
	{
		kernel_cache.index[i] = -1;
		kernel_cache.lru[i] = 0;
	}
	for (i = 0; i < totdoc; i++)
	{
		kernel_cache.occu[i] = 0;
		kernel_cache.invindex[i] = -1;
	}

	kernel_cache.activenum = totdoc;
	;
	for (i = 0; i < totdoc; i++)
	{
		kernel_cache.active2totdoc[i] = i;
		kernel_cache.totdoc2active[i] = i;
	}

	kernel_cache.time = 0;
}

void CachedKernel::get_kernel_row(
    int32_t docnum, int32_t* active2dnum, float64_t* buffer, bool full_line)
{
	int32_t i, j;
	KERNELCACHE_IDX start;

	int32_t num_vectors = kernel->get_num_vec_lhs();
	if (docnum >= num_vectors)
		docnum = 2 * num_vectors - 1 - docnum;

	/* is cached? */
	if (kernel_cache.index[docnum] != -1)
	{
		kernel_cache.lru[kernel_cache.index[docnum]] =
		    kernel_cache.time; /* lru */
		start = ((KERNELCACHE_IDX)kernel_cache.activenum) *
		        kernel_cache.index[docnum];

		if (full_line)
		{
			for (j = 0; j < kernel->get_num_vec_lhs(); j++)
			{
				if (kernel_cache.totdoc2active[j] >= 0)
					buffer[j] =
					    kernel_cache
					        .buffer[start + kernel_cache.totdoc2active[j]];
				else
					buffer[j] = (float64_t)kernel->kernel(docnum, j);
			}
		}
		else
		{
			for (i = 0; (j = active2dnum[i]) >= 0; i++)
			{
				if (kernel_cache.totdoc2active[j] >= 0)
					buffer[j] =
					    kernel_cache
					        .buffer[start + kernel_cache.totdoc2active[j]];
				else
				{
					int32_t k = j;
					if (k >= num_vectors)
						k = 2 * num_vectors - 1 - k;
					buffer[j] = (float64_t)kernel->kernel(docnum, k);
				}
			}
		}
	}
	else
	{
		if (full_line)
		{
			for (j = 0; j < kernel->get_num_vec_lhs(); j++)
				buffer[j] = (KERNELCACHE_ELEM)kernel->kernel(docnum, j);
		}
		else
		{
			for (i = 0; (j = active2dnum[i]) >= 0; i++)
			{
				int32_t k = j;
				if (k >= num_vectors)
					k = 2 * num_vectors - 1 - k;
				buffer[j] = (KERNELCACHE_ELEM)kernel->kernel(docnum, k);
			}
		}
	}
}

// Fills cache for the row m
void CachedKernel::cache_kernel_row(int32_t m)
{
	int32_t j, k, l;
	KERNELCACHE_ELEM* cache;

	int32_t num_vectors = kernel->get_num_vec_lhs();

	if (m >= num_vectors)
		m = 2 * num_vectors - 1 - m;

	if (!kernel_cache_check(m)) // not cached yet
	{
		cache = kernel_cache_clean_and_malloc(m);
		if (cache)
		{
			l = kernel_cache.totdoc2active[m];

			for (j = 0; j < kernel_cache.activenum; j++) // fill cache
			{
				k = kernel_cache.active2totdoc[j];

				if ((kernel_cache.index[k] != -1) && (l != -1) && (k != m))
				{
					cache[j] = kernel_cache.buffer
					               [((KERNELCACHE_IDX)kernel_cache.activenum) *
					                    kernel_cache.index[k] +
					                l];
				}
				else
				{
					if (k >= num_vectors)
						k = 2 * num_vectors - 1 - k;

					cache[j] = kernel->kernel(m, k);
				}
			}
		}
		else
			perror("Error: Kernel cache full! => increase cache size");
	}
}

void* CachedKernel::cache_multiple_kernel_row_helper(void* p)
{
	int32_t j, k, l;
	S_KTHREAD_PARAM* params = (S_KTHREAD_PARAM*)p;

	for (int32_t i = params->start; i < params->end; i++)
	{
		KERNELCACHE_ELEM* cache = params->cache[i];
		int32_t m = params->uncached_rows[i];
		l = params->kernel_cache->totdoc2active[m];

		for (j = 0; j < params->kernel_cache->activenum; j++) // fill cache
		{
			k = params->kernel_cache->active2totdoc[j];

			if ((params->kernel_cache->index[k] != -1) && (l != -1) &&
			    (!params->needs_computation[k]))
			{
				cache[j] =
				    params->kernel_cache->buffer
				        [((KERNELCACHE_IDX)params->kernel_cache->activenum) *
				             params->kernel_cache->index[k] +
				         l];
			}
			else
			{
				if (k >= params->num_vectors)
					k = 2 * params->num_vectors - 1 - k;

				cache[j] = params->kernel->kernel(m, k);
			}
		}

		// now line m is cached
		params->needs_computation[m] = 0;
	}
	return NULL;
}

// Fills cache for the rows in key
void CachedKernel::cache_multiple_kernel_rows(int32_t* rows, int32_t num_rows)
{
	int32_t nthreads = env()->get_num_threads();

	if (nthreads < 2)
	{
		for (int32_t i = 0; i < num_rows; i++)
			cache_kernel_row(rows[i]);
	}
	else
	{
		// fill up kernel cache
		int32_t* uncached_rows = SG_MALLOC(int32_t, num_rows);
		KERNELCACHE_ELEM** cache = SG_MALLOC(KERNELCACHE_ELEM*, num_rows);
		S_KTHREAD_PARAM params;
		int32_t num_threads = nthreads - 1;
		int32_t num_vec = kernel->get_num_vec_lhs();
		ASSERT(num_vec > 0)
		uint8_t* needs_computation = SG_CALLOC(uint8_t, num_vec);

		int32_t step = 0;
		int32_t num = 0;
		int32_t end = 0;

		// allocate cachelines if necessary
		for (int32_t i = 0; i < num_rows; i++)
		{
			int32_t idx = rows[i];
			if (idx >= num_vec)
				idx = 2 * num_vec - 1 - idx;

			if (kernel_cache_check(idx))
				continue;

			needs_computation[idx] = 1;
			uncached_rows[num] = idx;
			cache[num] = kernel_cache_clean_and_malloc(idx);

			if (!cache[num])
				SG_SERROR("Kernel cache full! => increase cache size\n")

			num++;
		}

		if (num > 0)
		{
			step = num / nthreads;

			if (step < 1)
			{
				num_threads = num - 1;
				step = 1;
			}

#pragma omp parallel for private(params)
			for (int32_t t = 0; t < num_threads; t++)
			{
				params.kernel = kernel;
				params.kernel_cache = &kernel_cache;
				params.cache = cache;
				params.uncached_rows = uncached_rows;
				params.needs_computation = needs_computation;
				params.num_uncached = num;
				params.start = t * step;
				params.end = (t + 1) * step;
				params.num_vectors = kernel->get_num_vec_lhs();
				end = params.end;

				cache_multiple_kernel_row_helper(&params);
			}
		}
		else
			num_threads = -1;

		S_KTHREAD_PARAM last_param;
		last_param.kernel = kernel;
		last_param.kernel_cache = &kernel_cache;
		last_param.cache = cache;
		last_param.uncached_rows = uncached_rows;
		last_param.needs_computation = needs_computation;
		last_param.start = end;
		last_param.num_uncached = num;
		last_param.end = num;
		last_param.num_vectors = kernel->get_num_vec_lhs();

		cache_multiple_kernel_row_helper(&last_param);

		SG_FREE(needs_computation);
		SG_FREE(cache);
		SG_FREE(uncached_rows);
	}
}

// remove numshrink columns in the cache
// which correspond to examples marked
void CachedKernel::kernel_cache_shrink(
    int32_t totdoc, int32_t numshrink, int32_t* after)
{
	ASSERT(totdoc > 0);
	int32_t i, j, jj, scount; // 0 in after.
	KERNELCACHE_IDX from = 0, to = 0;
	int32_t* keep;

	keep = SG_MALLOC(int32_t, totdoc);
	for (j = 0; j < totdoc; j++)
	{
		keep[j] = 1;
	}
	scount = 0;
	for (jj = 0; (jj < kernel_cache.activenum) && (scount < numshrink); jj++)
	{
		j = kernel_cache.active2totdoc[jj];
		if (!after[j])
		{
			scount++;
			keep[j] = 0;
		}
	}

	for (i = 0; i < kernel_cache.max_elems; i++)
	{
		for (jj = 0; jj < kernel_cache.activenum; jj++)
		{
			j = kernel_cache.active2totdoc[jj];
			if (!keep[j])
			{
				from++;
			}
			else
			{
				kernel_cache.buffer[to] = kernel_cache.buffer[from];
				to++;
				from++;
			}
		}
	}

	kernel_cache.activenum = 0;
	for (j = 0; j < totdoc; j++)
	{
		if ((keep[j]) && (kernel_cache.totdoc2active[j] != -1))
		{
			kernel_cache.active2totdoc[kernel_cache.activenum] = j;
			kernel_cache.totdoc2active[j] = kernel_cache.activenum;
			kernel_cache.activenum++;
		}
		else
		{
			kernel_cache.totdoc2active[j] = -1;
		}
	}

	kernel_cache.max_elems = (int32_t)kernel_cache.buffsize;

	if (kernel_cache.activenum > 0)
		kernel_cache.buffsize /= kernel_cache.activenum;

	if (kernel_cache.max_elems > totdoc)
		kernel_cache.max_elems = totdoc;

	SG_FREE(keep);
}

void CachedKernel::kernel_cache_reset_lru()
{
	int32_t maxlru = 0, k;

	for (k = 0; k < kernel_cache.max_elems; k++)
	{
		if (maxlru < kernel_cache.lru[k])
			maxlru = kernel_cache.lru[k];
	}
	for (k = 0; k < kernel_cache.max_elems; k++)
	{
		kernel_cache.lru[k] -= maxlru;
	}
}

void CachedKernel::kernel_cache_cleanup()
{
	SG_FREE(kernel_cache.index);
	SG_FREE(kernel_cache.occu);
	SG_FREE(kernel_cache.lru);
	SG_FREE(kernel_cache.invindex);
	SG_FREE(kernel_cache.active2totdoc);
	SG_FREE(kernel_cache.totdoc2active);
	SG_FREE(kernel_cache.buffer);
	memset(&kernel_cache, 0x0, sizeof(KERNEL_CACHE));
}

int32_t CachedKernel::kernel_cache_malloc()
{
	int32_t i;

	if (kernel_cache_space_available())
	{
		for (i = 0; i < kernel_cache.max_elems; i++)
		{
			if (!kernel_cache.occu[i])
			{
				kernel_cache.occu[i] = 1;
				kernel_cache.elems++;
				return (i);
			}
		}
	}
	return (-1);
}

void CachedKernel::kernel_cache_free(int32_t cacheidx)
{
	kernel_cache.occu[cacheidx] = 0;
	kernel_cache.elems--;
}

// remove least recently used cache
// element
int32_t CachedKernel::kernel_cache_free_lru()
{
	int32_t k, least_elem = -1, least_time;

	least_time = kernel_cache.time + 1;
	for (k = 0; k < kernel_cache.max_elems; k++)
	{
		if (kernel_cache.invindex[k] != -1)
		{
			if (kernel_cache.lru[k] < least_time)
			{
				least_time = kernel_cache.lru[k];
				least_elem = k;
			}
		}
	}

	if (least_elem != -1)
	{
		kernel_cache_free(least_elem);
		kernel_cache.index[kernel_cache.invindex[least_elem]] = -1;
		kernel_cache.invindex[least_elem] = -1;
		return (1);
	}
	return (0);
}

// Get a free cache entry. In case cache is full, the lru
// element is removed.
KERNELCACHE_ELEM* CachedKernel::kernel_cache_clean_and_malloc(int32_t cacheidx)
{
	int32_t result;
	if ((result = kernel_cache_malloc()) == -1)
	{
		if (kernel_cache_free_lru())
		{
			result = kernel_cache_malloc();
		}
	}
	kernel_cache.index[cacheidx] = result;
	if (result == -1)
	{
		return (0);
	}
	kernel_cache.invindex[result] = cacheidx;
	kernel_cache.lru[kernel_cache.index[cacheidx]] = kernel_cache.time; // lru
	return &kernel_cache.buffer
	            [((KERNELCACHE_IDX)kernel_cache.activenum) *
	             kernel_cache.index[cacheidx]];
}
#endif // USE_SVMLIGHT
