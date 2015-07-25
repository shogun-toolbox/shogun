/*
 * EXCEPT FOR THE KERNEL CACHING FUNCTIONS WHICH ARE (W) THORSTEN JOACHIMS
 * COPYRIGHT (C) 1999  UNIVERSITAET DORTMUND - ALL RIGHTS RESERVED
 *
 * this program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/io/File.h>
#include <shogun/lib/Time.h>
#include <shogun/lib/Signal.h>

#include <shogun/base/Parallel.h>

#include <shogun/kernel/Kernel.h>
#include <shogun/kernel/normalizer/IdentityKernelNormalizer.h>
#include <shogun/features/Features.h>
#include <shogun/base/Parameter.h>

#include <shogun/classifier/svm/SVM.h>

#include <string.h>
#include <unistd.h>
#include <shogun/mathematics/Math.h>

#ifdef HAVE_PTHREAD
#include <pthread.h>
#endif

using namespace shogun;

CKernel::CKernel() : CSGObject()
{
	init();
	register_params();
}

CKernel::CKernel(int32_t size) : CSGObject()
{
	init();

	if (size<10)
		size=10;

	cache_size=size;
	register_params();
}


CKernel::CKernel(CFeatures* p_lhs, CFeatures* p_rhs, int32_t size) : CSGObject()
{
	init();

	if (size<10)
		size=10;

	cache_size=size;

	set_normalizer(new CIdentityKernelNormalizer());
	init(p_lhs, p_rhs);
	register_params();
}

CKernel::~CKernel()
{
	if (get_is_initialized())
		SG_ERROR("Kernel still initialized on destruction.\n")

	remove_lhs_and_rhs();
	SG_UNREF(normalizer);

	SG_INFO("Kernel deleted (%p).\n", this)
}

#ifdef USE_SVMLIGHT
void CKernel::resize_kernel_cache(KERNELCACHE_IDX size, bool regression_hack)
{
	if (size<10)
		size=10;

	kernel_cache_cleanup();
	cache_size=size;

	if (has_features() && get_num_vec_lhs())
		kernel_cache_init(cache_size, regression_hack);
}
#endif //USE_SVMLIGHT

bool CKernel::init(CFeatures* l, CFeatures* r)
{
	/* make sure that features are not deleted if same ones are used */
	SG_REF(l);
	SG_REF(r);

	//make sure features were indeed supplied
	REQUIRE(l, "CKernel::init(%p, %p): Left hand side features required!\n", l, r)
	REQUIRE(r, "CKernel::init(%p, %p): Right hand side features required!\n", l, r)

	//make sure features are compatible
	if (l->support_compatible_class())
	{
		REQUIRE(l->get_feature_class_compatibility(r->get_feature_class()),
			"Right hand side of features (%s) must be compatible with left hand side features (%s)\n",
			l->get_name(), r->get_name());
	}
	else
	{
		REQUIRE(l->get_feature_class()==r->get_feature_class(),
			"Right hand side of features (%s) must be compatible with left hand side features (%s)\n",
			l->get_name(), r->get_name())
	}
	ASSERT(l->get_feature_type()==r->get_feature_type())

	//remove references to previous features
	remove_lhs_and_rhs();

    //increase reference counts
    SG_REF(l);
    if (l==r)
		lhs_equals_rhs=true;
	else // l!=r
        SG_REF(r);

	lhs=l;
	rhs=r;

	ASSERT(!num_lhs || num_lhs==l->get_num_vectors())
	ASSERT(!num_rhs || num_rhs==l->get_num_vectors())

	num_lhs=l->get_num_vectors();
	num_rhs=r->get_num_vectors();

	/* unref "safety" refs from beginning */
	SG_UNREF(r);
	SG_UNREF(l);

	SG_DEBUG("leaving CKernel::init(%p, %p)\n", l, r)
	return true;
}

bool CKernel::set_normalizer(CKernelNormalizer* n)
{
	SG_REF(n);
	if (lhs && rhs)
		n->init(this);

	SG_UNREF(normalizer);
	normalizer=n;

	return (normalizer!=NULL);
}

CKernelNormalizer* CKernel::get_normalizer()
{
	SG_REF(normalizer)
	return normalizer;
}

bool CKernel::init_normalizer()
{
	return normalizer->init(this);
}

void CKernel::cleanup()
{
	remove_lhs_and_rhs();
}

#ifdef USE_SVMLIGHT
/****************************** Cache handling *******************************/

void CKernel::kernel_cache_init(int32_t buffsize, bool regression_hack)
{
	int32_t totdoc=get_num_vec_lhs();
	if (totdoc<=0)
	{
		SG_ERROR("kernel has zero rows: num_lhs=%d num_rhs=%d\n",
				get_num_vec_lhs(), get_num_vec_rhs());
	}
	uint64_t buffer_size=0;
	int32_t i;

	//in regression the additional constraints are made by doubling the training data
	if (regression_hack)
		totdoc*=2;

	buffer_size=((uint64_t) buffsize)*1024*1024/sizeof(KERNELCACHE_ELEM);
	if (buffer_size>((uint64_t) totdoc)*totdoc)
		buffer_size=((uint64_t) totdoc)*totdoc;

	SG_INFO("using a kernel cache of size %lld MB (%lld bytes) for %s Kernel\n", buffer_size*sizeof(KERNELCACHE_ELEM)/1024/1024, buffer_size*sizeof(KERNELCACHE_ELEM), get_name())

	//make sure it fits in the *signed* KERNELCACHE_IDX type
	ASSERT(buffer_size < (((uint64_t) 1) << (sizeof(KERNELCACHE_IDX)*8-1)))

	kernel_cache.index = SG_MALLOC(int32_t, totdoc);
	kernel_cache.occu = SG_MALLOC(int32_t, totdoc);
	kernel_cache.lru = SG_MALLOC(int32_t, totdoc);
	kernel_cache.invindex = SG_MALLOC(int32_t, totdoc);
	kernel_cache.active2totdoc = SG_MALLOC(int32_t, totdoc);
	kernel_cache.totdoc2active = SG_MALLOC(int32_t, totdoc);
	kernel_cache.buffer = SG_MALLOC(KERNELCACHE_ELEM, buffer_size);
	kernel_cache.buffsize=buffer_size;
	kernel_cache.max_elems=(int32_t) (kernel_cache.buffsize/totdoc);

	if(kernel_cache.max_elems>totdoc) {
		kernel_cache.max_elems=totdoc;
	}

	kernel_cache.elems=0;   // initialize cache
	for(i=0;i<totdoc;i++) {
		kernel_cache.index[i]=-1;
		kernel_cache.lru[i]=0;
	}
	for(i=0;i<totdoc;i++) {
		kernel_cache.occu[i]=0;
		kernel_cache.invindex[i]=-1;
	}

	kernel_cache.activenum=totdoc;;
	for(i=0;i<totdoc;i++) {
		kernel_cache.active2totdoc[i]=i;
		kernel_cache.totdoc2active[i]=i;
	}

	kernel_cache.time=0;
}

void CKernel::get_kernel_row(
	int32_t docnum, int32_t *active2dnum, float64_t *buffer, bool full_line)
{
	int32_t i,j;
	KERNELCACHE_IDX start;

	int32_t num_vectors = get_num_vec_lhs();
	if (docnum>=num_vectors)
		docnum=2*num_vectors-1-docnum;

	/* is cached? */
	if(kernel_cache.index[docnum] != -1)
	{
		kernel_cache.lru[kernel_cache.index[docnum]]=kernel_cache.time; /* lru */
		start=((KERNELCACHE_IDX) kernel_cache.activenum)*kernel_cache.index[docnum];

		if (full_line)
		{
			for(j=0;j<get_num_vec_lhs();j++)
			{
				if(kernel_cache.totdoc2active[j] >= 0)
					buffer[j]=kernel_cache.buffer[start+kernel_cache.totdoc2active[j]];
				else
					buffer[j]=(float64_t) kernel(docnum, j);
			}
		}
		else
		{
			for(i=0;(j=active2dnum[i])>=0;i++)
			{
				if(kernel_cache.totdoc2active[j] >= 0)
					buffer[j]=kernel_cache.buffer[start+kernel_cache.totdoc2active[j]];
				else
				{
					int32_t k=j;
					if (k>=num_vectors)
						k=2*num_vectors-1-k;
					buffer[j]=(float64_t) kernel(docnum, k);
				}
			}
		}
	}
	else
	{
		if (full_line)
		{
			for(j=0;j<get_num_vec_lhs();j++)
				buffer[j]=(KERNELCACHE_ELEM) kernel(docnum, j);
		}
		else
		{
			for(i=0;(j=active2dnum[i])>=0;i++)
			{
				int32_t k=j;
				if (k>=num_vectors)
					k=2*num_vectors-1-k;
				buffer[j]=(KERNELCACHE_ELEM) kernel(docnum, k);
			}
		}
	}
}


// Fills cache for the row m
void CKernel::cache_kernel_row(int32_t m)
{
	register int32_t j,k,l;
	register KERNELCACHE_ELEM *cache;

	int32_t num_vectors = get_num_vec_lhs();

	if (m>=num_vectors)
		m=2*num_vectors-1-m;

	if(!kernel_cache_check(m))   // not cached yet
	{
		cache = kernel_cache_clean_and_malloc(m);
		if(cache) {
			l=kernel_cache.totdoc2active[m];

			for(j=0;j<kernel_cache.activenum;j++)  // fill cache
			{
				k=kernel_cache.active2totdoc[j];

				if((kernel_cache.index[k] != -1) && (l != -1) && (k != m)) {
					cache[j]=kernel_cache.buffer[((KERNELCACHE_IDX) kernel_cache.activenum)
						*kernel_cache.index[k]+l];
				}
				else
				{
					if (k>=num_vectors)
						k=2*num_vectors-1-k;

					cache[j]=kernel(m, k);
				}
			}
		}
		else
			perror("Error: Kernel cache full! => increase cache size");
	}
}


void* CKernel::cache_multiple_kernel_row_helper(void* p)
{
	int32_t j,k,l;
	S_KTHREAD_PARAM* params = (S_KTHREAD_PARAM*) p;

	for (int32_t i=params->start; i<params->end; i++)
	{
		KERNELCACHE_ELEM* cache=params->cache[i];
		int32_t m = params->uncached_rows[i];
		l=params->kernel_cache->totdoc2active[m];

		for(j=0;j<params->kernel_cache->activenum;j++)  // fill cache
		{
			k=params->kernel_cache->active2totdoc[j];

			if((params->kernel_cache->index[k] != -1) && (l != -1) && (!params->needs_computation[k])) {
				cache[j]=params->kernel_cache->buffer[((KERNELCACHE_IDX) params->kernel_cache->activenum)
					*params->kernel_cache->index[k]+l];
			}
			else
				{
					if (k>=params->num_vectors)
						k=2*params->num_vectors-1-k;

					cache[j]=params->kernel->kernel(m, k);
				}
		}

		//now line m is cached
		params->needs_computation[m]=0;
	}
	return NULL;
}

// Fills cache for the rows in key
void CKernel::cache_multiple_kernel_rows(int32_t* rows, int32_t num_rows)
{
#ifdef HAVE_PTHREAD
	int32_t nthreads=parallel->get_num_threads();

	if (nthreads<2)
	{
#endif
		for(int32_t i=0;i<num_rows;i++)
			cache_kernel_row(rows[i]);
#ifdef HAVE_PTHREAD
	}
	else
	{
		// fill up kernel cache
		int32_t* uncached_rows = SG_MALLOC(int32_t, num_rows);
		KERNELCACHE_ELEM** cache = SG_MALLOC(KERNELCACHE_ELEM*, num_rows);
		pthread_t* threads = SG_MALLOC(pthread_t, nthreads-1);
		S_KTHREAD_PARAM* params = SG_MALLOC(S_KTHREAD_PARAM, nthreads-1);
		int32_t num_threads=nthreads-1;
		int32_t num_vec=get_num_vec_lhs();
		ASSERT(num_vec>0)
		uint8_t* needs_computation=SG_CALLOC(uint8_t, num_vec);

		int32_t step=0;
		int32_t num=0;
		int32_t end=0;

		// allocate cachelines if necessary
		for (int32_t i=0; i<num_rows; i++)
		{
			int32_t idx=rows[i];
			if (idx>=num_vec)
				idx=2*num_vec-1-idx;

			if (kernel_cache_check(idx))
				continue;

			needs_computation[idx]=1;
			uncached_rows[num]=idx;
			cache[num]= kernel_cache_clean_and_malloc(idx);

			if (!cache[num])
				SG_ERROR("Kernel cache full! => increase cache size\n")

			num++;
		}

		if (num>0)
		{
			step= num/nthreads;

			if (step<1)
			{
				num_threads=num-1;
				step=1;
			}

			for (int32_t t=0; t<num_threads; t++)
			{
				params[t].kernel = this;
				params[t].kernel_cache = &kernel_cache;
				params[t].cache = cache;
				params[t].uncached_rows = uncached_rows;
				params[t].needs_computation = needs_computation;
				params[t].num_uncached = num;
				params[t].start = t*step;
				params[t].end = (t+1)*step;
				params[t].num_vectors = get_num_vec_lhs();
				end=params[t].end;

				int code=pthread_create(&threads[t], NULL,
						CKernel::cache_multiple_kernel_row_helper, (void*)&params[t]);

				if (code != 0)
				{
					SG_WARNING("Thread creation failed (thread %d of %d) "
							"with error:'%s'\n",t, num_threads, strerror(code));
					num_threads=t;
					end=t*step;
					break;
				}
			}
		}
		else
			num_threads=-1;


		S_KTHREAD_PARAM last_param;
		last_param.kernel = this;
		last_param.kernel_cache = &kernel_cache;
		last_param.cache = cache;
		last_param.uncached_rows = uncached_rows;
		last_param.needs_computation = needs_computation;
		last_param.start = end;
		last_param.num_uncached = num;
		last_param.end = num;
		last_param.num_vectors = get_num_vec_lhs();

		cache_multiple_kernel_row_helper(&last_param);


		for (int32_t t=0; t<num_threads; t++)
		{
			if (pthread_join(threads[t], NULL) != 0)
				SG_WARNING("pthread_join of thread %d/%d failed\n", t, num_threads)
		}

		SG_FREE(needs_computation);
		SG_FREE(params);
		SG_FREE(threads);
		SG_FREE(cache);
		SG_FREE(uncached_rows);
	}
#endif
}

// remove numshrink columns in the cache
// which correspond to examples marked
void CKernel::kernel_cache_shrink(
	int32_t totdoc, int32_t numshrink, int32_t *after)
{
	ASSERT(totdoc > 0);
	register int32_t i,j,jj,scount;     // 0 in after.
	KERNELCACHE_IDX from=0,to=0;
	int32_t *keep;

	keep=SG_MALLOC(int32_t, totdoc);
	for(j=0;j<totdoc;j++) {
		keep[j]=1;
	}
	scount=0;
	for(jj=0;(jj<kernel_cache.activenum) && (scount<numshrink);jj++) {
		j=kernel_cache.active2totdoc[jj];
		if(!after[j]) {
			scount++;
			keep[j]=0;
		}
	}

	for(i=0;i<kernel_cache.max_elems;i++) {
		for(jj=0;jj<kernel_cache.activenum;jj++) {
			j=kernel_cache.active2totdoc[jj];
			if(!keep[j]) {
				from++;
			}
			else {
				kernel_cache.buffer[to]=kernel_cache.buffer[from];
				to++;
				from++;
			}
		}
	}

	kernel_cache.activenum=0;
	for(j=0;j<totdoc;j++) {
		if((keep[j]) && (kernel_cache.totdoc2active[j] != -1)) {
			kernel_cache.active2totdoc[kernel_cache.activenum]=j;
			kernel_cache.totdoc2active[j]=kernel_cache.activenum;
			kernel_cache.activenum++;
		}
		else {
			kernel_cache.totdoc2active[j]=-1;
		}
	}

	kernel_cache.max_elems= (int32_t) kernel_cache.buffsize;

	if (kernel_cache.activenum>0)
		kernel_cache.buffsize/=kernel_cache.activenum;

	if(kernel_cache.max_elems>totdoc)
		kernel_cache.max_elems=totdoc;

	SG_FREE(keep);

}

void CKernel::kernel_cache_reset_lru()
{
	int32_t maxlru=0,k;

	for(k=0;k<kernel_cache.max_elems;k++) {
		if(maxlru < kernel_cache.lru[k])
			maxlru=kernel_cache.lru[k];
	}
	for(k=0;k<kernel_cache.max_elems;k++) {
		kernel_cache.lru[k]-=maxlru;
	}
}

void CKernel::kernel_cache_cleanup()
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

int32_t CKernel::kernel_cache_malloc()
{
  int32_t i;

  if(kernel_cache_space_available()) {
    for(i=0;i<kernel_cache.max_elems;i++) {
      if(!kernel_cache.occu[i]) {
	kernel_cache.occu[i]=1;
	kernel_cache.elems++;
	return(i);
      }
    }
  }
  return(-1);
}

void CKernel::kernel_cache_free(int32_t cacheidx)
{
	kernel_cache.occu[cacheidx]=0;
	kernel_cache.elems--;
}

// remove least recently used cache
// element
int32_t CKernel::kernel_cache_free_lru()
{
  register int32_t k,least_elem=-1,least_time;

  least_time=kernel_cache.time+1;
  for(k=0;k<kernel_cache.max_elems;k++) {
    if(kernel_cache.invindex[k] != -1) {
      if(kernel_cache.lru[k]<least_time) {
	least_time=kernel_cache.lru[k];
	least_elem=k;
      }
    }
  }

  if(least_elem != -1) {
    kernel_cache_free(least_elem);
    kernel_cache.index[kernel_cache.invindex[least_elem]]=-1;
    kernel_cache.invindex[least_elem]=-1;
    return(1);
  }
  return(0);
}

// Get a free cache entry. In case cache is full, the lru
// element is removed.
KERNELCACHE_ELEM* CKernel::kernel_cache_clean_and_malloc(int32_t cacheidx)
{
	int32_t result;
	if((result = kernel_cache_malloc()) == -1) {
		if(kernel_cache_free_lru()) {
			result = kernel_cache_malloc();
		}
	}
	kernel_cache.index[cacheidx]=result;
	if(result == -1) {
		return(0);
	}
	kernel_cache.invindex[result]=cacheidx;
	kernel_cache.lru[kernel_cache.index[cacheidx]]=kernel_cache.time; // lru
	return &kernel_cache.buffer[((KERNELCACHE_IDX) kernel_cache.activenum)*kernel_cache.index[cacheidx]];
}
#endif //USE_SVMLIGHT

void CKernel::load(CFile* loader)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
}

void CKernel::save(CFile* writer)
{
	SGMatrix<float64_t> k_matrix=get_kernel_matrix<float64_t>();
	SG_SET_LOCALE_C;
	writer->set_matrix(k_matrix.matrix, k_matrix.num_rows, k_matrix.num_cols);
	SG_RESET_LOCALE;
}

void CKernel::remove_lhs_and_rhs()
{
	SG_DEBUG("entering CKernel::remove_lhs_and_rhs\n")
	if (rhs!=lhs)
		SG_UNREF(rhs);
	rhs = NULL;
	num_rhs=0;

	SG_UNREF(lhs);
	lhs = NULL;
	num_lhs=0;
	lhs_equals_rhs=false;

#ifdef USE_SVMLIGHT
	cache_reset();
#endif //USE_SVMLIGHT
	SG_DEBUG("leaving CKernel::remove_lhs_and_rhs\n")
}

void CKernel::remove_lhs()
{
	if (rhs==lhs)
		rhs=NULL;
	SG_UNREF(lhs);
	lhs = NULL;
	num_lhs=0;
	lhs_equals_rhs=false;
#ifdef USE_SVMLIGHT
	cache_reset();
#endif //USE_SVMLIGHT
}

/// takes all necessary steps if the rhs is removed from kernel
void CKernel::remove_rhs()
{
	if (rhs!=lhs)
		SG_UNREF(rhs);
	rhs = NULL;
	num_rhs=0;
	lhs_equals_rhs=false;

#ifdef USE_SVMLIGHT
	cache_reset();
#endif //USE_SVMLIGHT
}

#define ENUM_CASE(n) case n: SG_INFO(#n " ") break;

void CKernel::list_kernel()
{
	SG_INFO("%p - \"%s\" weight=%1.2f OPT:%s", this, get_name(),
			get_combined_kernel_weight(),
			get_optimization_type()==FASTBUTMEMHUNGRY ? "FASTBUTMEMHUNGRY" :
			"SLOWBUTMEMEFFICIENT");

	switch (get_kernel_type())
	{
		ENUM_CASE(K_UNKNOWN)
		ENUM_CASE(K_LINEAR)
		ENUM_CASE(K_POLY)
		ENUM_CASE(K_GAUSSIAN)
		ENUM_CASE(K_GAUSSIANSHIFT)
		ENUM_CASE(K_GAUSSIANMATCH)
		ENUM_CASE(K_HISTOGRAM)
		ENUM_CASE(K_SALZBERG)
		ENUM_CASE(K_LOCALITYIMPROVED)
		ENUM_CASE(K_SIMPLELOCALITYIMPROVED)
		ENUM_CASE(K_FIXEDDEGREE)
		ENUM_CASE(K_WEIGHTEDDEGREE)
		ENUM_CASE(K_WEIGHTEDDEGREEPOS)
		ENUM_CASE(K_WEIGHTEDDEGREERBF)
		ENUM_CASE(K_WEIGHTEDCOMMWORDSTRING)
		ENUM_CASE(K_POLYMATCH)
		ENUM_CASE(K_ALIGNMENT)
		ENUM_CASE(K_COMMWORDSTRING)
		ENUM_CASE(K_COMMULONGSTRING)
		ENUM_CASE(K_SPECTRUMRBF)
		ENUM_CASE(K_COMBINED)
		ENUM_CASE(K_AUC)
		ENUM_CASE(K_CUSTOM)
		ENUM_CASE(K_SIGMOID)
		ENUM_CASE(K_CHI2)
		ENUM_CASE(K_DIAG)
		ENUM_CASE(K_CONST)
		ENUM_CASE(K_DISTANCE)
		ENUM_CASE(K_LOCALALIGNMENT)
		ENUM_CASE(K_PYRAMIDCHI2)
		ENUM_CASE(K_OLIGO)
		ENUM_CASE(K_MATCHWORD)
		ENUM_CASE(K_TPPK)
		ENUM_CASE(K_REGULATORYMODULES)
		ENUM_CASE(K_SPARSESPATIALSAMPLE)
		ENUM_CASE(K_HISTOGRAMINTERSECTION)
		ENUM_CASE(K_WAVELET)
		ENUM_CASE(K_WAVE)
		ENUM_CASE(K_CAUCHY)
		ENUM_CASE(K_TSTUDENT)
		ENUM_CASE(K_MULTIQUADRIC)
		ENUM_CASE(K_EXPONENTIAL)
		ENUM_CASE(K_RATIONAL_QUADRATIC)
		ENUM_CASE(K_POWER)
		ENUM_CASE(K_SPHERICAL)
		ENUM_CASE(K_LOG)
		ENUM_CASE(K_SPLINE)
		ENUM_CASE(K_ANOVA)
		ENUM_CASE(K_CIRCULAR)
		ENUM_CASE(K_INVERSEMULTIQUADRIC)
		ENUM_CASE(K_SPECTRUMMISMATCHRBF)
		ENUM_CASE(K_DISTANTSEGMENTS)
		ENUM_CASE(K_BESSEL)
		ENUM_CASE(K_JENSENSHANNON)
		ENUM_CASE(K_DIRECTOR)
		ENUM_CASE(K_PRODUCT)
		ENUM_CASE(K_EXPONENTIALARD)
		ENUM_CASE(K_GAUSSIANARD)
		ENUM_CASE(K_GAUSSIANARDSPARSE)
		ENUM_CASE(K_STREAMING)
		ENUM_CASE(K_PERIODIC)
	}

	switch (get_feature_class())
	{
		ENUM_CASE(C_UNKNOWN)
		ENUM_CASE(C_DENSE)
		ENUM_CASE(C_SPARSE)
		ENUM_CASE(C_STRING)
		ENUM_CASE(C_STREAMING_DENSE)
		ENUM_CASE(C_STREAMING_SPARSE)
		ENUM_CASE(C_STREAMING_STRING)
		ENUM_CASE(C_STREAMING_VW)
		ENUM_CASE(C_COMBINED)
		ENUM_CASE(C_COMBINED_DOT)
		ENUM_CASE(C_WD)
		ENUM_CASE(C_SPEC)
		ENUM_CASE(C_WEIGHTEDSPEC)
		ENUM_CASE(C_POLY)
		ENUM_CASE(C_BINNED_DOT)
		ENUM_CASE(C_DIRECTOR_DOT)
		ENUM_CASE(C_LATENT)
		ENUM_CASE(C_MATRIX)
		ENUM_CASE(C_FACTOR_GRAPH)
		ENUM_CASE(C_INDEX)
		ENUM_CASE(C_SUB_SAMPLES_DENSE)
		ENUM_CASE(C_ANY)
	}

	switch (get_feature_type())
	{
		ENUM_CASE(F_UNKNOWN)
		ENUM_CASE(F_BOOL)
		ENUM_CASE(F_CHAR)
		ENUM_CASE(F_BYTE)
		ENUM_CASE(F_SHORT)
		ENUM_CASE(F_WORD)
		ENUM_CASE(F_INT)
		ENUM_CASE(F_UINT)
		ENUM_CASE(F_LONG)
		ENUM_CASE(F_ULONG)
		ENUM_CASE(F_SHORTREAL)
		ENUM_CASE(F_DREAL)
		ENUM_CASE(F_LONGREAL)
		ENUM_CASE(F_ANY)
	}
	SG_INFO("\n")
}
#undef ENUM_CASE

bool CKernel::init_optimization(
	int32_t count, int32_t *IDX, float64_t * weights)
{
   SG_ERROR("kernel does not support linadd optimization\n")
	return false ;
}

bool CKernel::delete_optimization()
{
   SG_ERROR("kernel does not support linadd optimization\n")
	return false;
}

float64_t CKernel::compute_optimized(int32_t vector_idx)
{
   SG_ERROR("kernel does not support linadd optimization\n")
	return 0;
}

void CKernel::compute_batch(
	int32_t num_vec, int32_t* vec_idx, float64_t* target, int32_t num_suppvec,
	int32_t* IDX, float64_t* weights, float64_t factor)
{
   SG_ERROR("kernel does not support batch computation\n")
}

void CKernel::add_to_normal(int32_t vector_idx, float64_t weight)
{
   SG_ERROR("kernel does not support linadd optimization, add_to_normal not implemented\n")
}

void CKernel::clear_normal()
{
   SG_ERROR("kernel does not support linadd optimization, clear_normal not implemented\n")
}

int32_t CKernel::get_num_subkernels()
{
	return 1;
}

void CKernel::compute_by_subkernel(
	int32_t vector_idx, float64_t * subkernel_contrib)
{
   SG_ERROR("kernel compute_by_subkernel not implemented\n")
}

const float64_t* CKernel::get_subkernel_weights(int32_t &num_weights)
{
	num_weights=1 ;
	return &combined_kernel_weight ;
}

SGVector<float64_t> CKernel::get_subkernel_weights()
{
	int num_weights = 1;
	const float64_t* weight = get_subkernel_weights(num_weights);
	return SGVector<float64_t>(const_cast<float64_t*>(weight),1,false);
}

void CKernel::set_subkernel_weights(const SGVector<float64_t> weights)
{
	ASSERT(weights.vector)
	if (weights.vlen!=1)
      SG_ERROR("number of subkernel weights should be one ...\n")

	combined_kernel_weight = weights.vector[0] ;
}

CKernel* CKernel::obtain_from_generic(CSGObject* kernel)
{
	if (kernel)
	{
		CKernel* casted=dynamic_cast<CKernel*>(kernel);
		REQUIRE(casted, "CKernel::obtain_from_generic(): Error, provided object"
				" of class \"%s\" is not a subclass of CKernel!\n",
				kernel->get_name());
		return casted;
	}
	else
		return NULL;
}

bool CKernel::init_optimization_svm(CSVM * svm)
{
	int32_t num_suppvec=svm->get_num_support_vectors();
	int32_t* sv_idx=SG_MALLOC(int32_t, num_suppvec);
	float64_t* sv_weight=SG_MALLOC(float64_t, num_suppvec);

	for (int32_t i=0; i<num_suppvec; i++)
	{
		sv_idx[i]    = svm->get_support_vector(i);
		sv_weight[i] = svm->get_alpha(i);
	}
	bool ret = init_optimization(num_suppvec, sv_idx, sv_weight);

	SG_FREE(sv_idx);
	SG_FREE(sv_weight);
	return ret;
}

void CKernel::load_serializable_post() throw (ShogunException)
{
	CSGObject::load_serializable_post();
	if (lhs_equals_rhs)
		rhs=lhs;
}

void CKernel::save_serializable_pre() throw (ShogunException)
{
	CSGObject::save_serializable_pre();

	if (lhs_equals_rhs)
		rhs=NULL;
}

void CKernel::save_serializable_post() throw (ShogunException)
{
	CSGObject::save_serializable_post();

	if (lhs_equals_rhs)
		rhs=lhs;
}

void CKernel::register_params()   {
	SG_ADD(&cache_size, "cache_size",
	    "Cache size in MB.", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**) &lhs, "lhs",
      "Feature vectors to occur on left hand side.", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**) &rhs, "rhs",
      "Feature vectors to occur on right hand side.", MS_NOT_AVAILABLE);
	SG_ADD(&lhs_equals_rhs, "lhs_equals_rhs",
	    "If features on lhs are the same as on rhs.", MS_NOT_AVAILABLE);
	SG_ADD(&num_lhs, "num_lhs", "Number of feature vectors on left hand side.",
	    MS_NOT_AVAILABLE);
	SG_ADD(&num_rhs, "num_rhs", "Number of feature vectors on right hand side.",
	    MS_NOT_AVAILABLE);
	SG_ADD(&combined_kernel_weight, "combined_kernel_weight",
			"Combined kernel weight.", MS_AVAILABLE);
	SG_ADD(&optimization_initialized, "optimization_initialized",
		  "Optimization is initialized.", MS_NOT_AVAILABLE);
	SG_ADD((machine_int_t*) &opt_type, "opt_type",
		  "Optimization type.", MS_NOT_AVAILABLE);
	SG_ADD(&properties, "properties", "Kernel properties.", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**) &normalizer, "normalizer", "Normalize the kernel.",
	    MS_AVAILABLE);
}


void CKernel::init()
{
	cache_size=10;
	kernel_matrix=NULL;
	lhs=NULL;
	rhs=NULL;
	num_lhs=0;
	num_rhs=0;
	lhs_equals_rhs=false;
	combined_kernel_weight=1;
	optimization_initialized=false;
	opt_type=FASTBUTMEMHUNGRY;
	properties=KP_NONE;
	normalizer=NULL;

#ifdef USE_SVMLIGHT
	memset(&kernel_cache, 0x0, sizeof(KERNEL_CACHE));
#endif //USE_SVMLIGHT

	set_normalizer(new CIdentityKernelNormalizer());
}

namespace shogun
{
/** kernel thread parameters */
template <class T> struct K_THREAD_PARAM
{
	/** kernel */
	CKernel* kernel;
	/** start (unit row) */
	int32_t start;
	/** end (unit row) */
	int32_t end;
	/** start (unit number of elements) */
	int64_t total_start;
	/** end (unit number of elements) */
	int64_t total_end;
	/** m */
	int32_t m;
	/** n */
	int32_t n;
	/** result */
	T* result;
	/** kernel matrix k(i,j)=k(j,i) */
	bool symmetric;
	/** output progress */
	bool verbose;
};
}

float64_t CKernel::sum_symmetric_block(index_t block_begin, index_t block_size,
		bool no_diag)
{
	SG_DEBUG("Entering\n");

	REQUIRE(has_features(), "No features assigned to kernel\n")
	REQUIRE(lhs_equals_rhs, "The kernel matrix is not symmetric!\n")
	REQUIRE(block_begin>=0 && block_begin<num_rhs,
			"Invalid block begin index (%d, %d)!\n", block_begin, block_begin)
	REQUIRE(block_begin+block_size<=num_rhs,
			"Invalid block size (%d) at starting index (%d, %d)! "
			"Please use smaller blocks!", block_size, block_begin, block_begin)
	REQUIRE(block_size>=1, "Invalid block size (%d)!\n", block_size)

	float64_t sum=0.0;

	// since the block is symmetric with main diagonal inside, we can save half
	// the computation with using only the upper triangular part.
	// this can be done in parallel
#pragma omp parallel for
	for (index_t i=0; i<block_size; ++i)
	{
		// compute the kernel values on the upper triangular part of the kernel
		// matrix and compute sum on the fly
		for (index_t j=i+1; j<block_size; ++j)
		{
			float64_t k=kernel(i+block_begin, j+block_begin);
#pragma omp atomic
			sum+=k;
		}
	}

	// the actual sum would be twice of what we computed
	sum*=2;

	// add the diagonal elements if required - keeping this check
	// outside of the loop to save cycles
	if (!no_diag)
	{
#pragma omp parallel for
		for (index_t i=0; i<block_size; ++i)
		{
			float64_t diag=kernel(i+block_begin, i+block_begin);
#pragma omp atomic
			sum+=diag;
		}
	}

	SG_DEBUG("Leaving\n");

	return sum;
}

float64_t CKernel::sum_block(index_t block_begin_row, index_t block_begin_col,
		index_t block_size_row, index_t block_size_col, bool no_diag)
{
	SG_DEBUG("Entering\n");

	REQUIRE(has_features(), "No features assigned to kernel\n")
	REQUIRE(block_begin_row>=0 && block_begin_row<num_lhs &&
			block_begin_col>=0 && block_begin_col<num_rhs,
			"Invalid block begin index (%d, %d)!\n",
			block_begin_row, block_begin_col)
	REQUIRE(block_begin_row+block_size_row<=num_lhs &&
			block_begin_col+block_size_col<=num_rhs,
			"Invalid block size (%d, %d) at starting index (%d, %d)! "
			"Please use smaller blocks!", block_size_row, block_size_col,
			block_begin_row, block_begin_col)
	REQUIRE(block_size_row>=1 && block_size_col>=1,
			"Invalid block size (%d, %d)!\n", block_size_row, block_size_col)

	// check if removal of diagonal is required/valid
	if (no_diag && block_size_row!=block_size_col)
	{
		SG_WARNING("Not removing the main diagonal since block is not square!\n");
		no_diag=false;
	}

	float64_t sum=0.0;

	// this can be done in parallel for the rows/cols
#pragma omp parallel for
	for (index_t i=0; i<block_size_row; ++i)
	{
		// compute the kernel values and compute sum on the fly
		for (index_t j=0; j<block_size_col; ++j)
		{
			float64_t k=no_diag && i==j ? 0 :
				kernel(i+block_begin_row, j+block_begin_col);
#pragma omp atomic
			sum+=k;
		}
	}

	SG_DEBUG("Leaving\n");

	return sum;
}

SGVector<float64_t> CKernel::row_wise_sum_symmetric_block(index_t block_begin,
		index_t block_size, bool no_diag)
{
	SG_DEBUG("Entering\n");

	REQUIRE(has_features(), "No features assigned to kernel\n")
	REQUIRE(lhs_equals_rhs, "The kernel matrix is not symmetric!\n")
	REQUIRE(block_begin>=0 && block_begin<num_rhs,
			"Invalid block begin index (%d, %d)!\n", block_begin, block_begin)
	REQUIRE(block_begin+block_size<=num_rhs,
			"Invalid block size (%d) at starting index (%d, %d)! "
			"Please use smaller blocks!", block_size, block_begin, block_begin)
	REQUIRE(block_size>=1, "Invalid block size (%d)!\n", block_size)

	// initialize the vector that accumulates the row/col-wise sum on the go
	SGVector<float64_t> row_sum(block_size);
	row_sum.set_const(0.0);

	// since the block is symmetric with main diagonal inside, we can save half
	// the computation with using only the upper triangular part.
	// this can be done in parallel for the rows/cols
#pragma omp parallel for
	for (index_t i=0; i<block_size; ++i)
	{
		// compute the kernel values on the upper triangular part of the kernel
		// matrix and compute row-wise sum on the fly
		for (index_t j=i+1; j<block_size; ++j)
		{
			float64_t k=kernel(i+block_begin, j+block_begin);
#pragma omp critical
			{
				row_sum[i]+=k;
				row_sum[j]+=k;
			}
		}
	}

	// add the diagonal elements if required - keeping this check
	// outside of the loop to save cycles
	if (!no_diag)
	{
#pragma omp parallel for
		for (index_t i=0; i<block_size; ++i)
		{
			float64_t diag=kernel(i+block_begin, i+block_begin);
			row_sum[i]+=diag;
		}
	}

	SG_DEBUG("Leaving\n");

	return row_sum;
}

SGMatrix<float64_t> CKernel::row_wise_sum_squared_sum_symmetric_block(index_t
		block_begin, index_t block_size, bool no_diag)
{
	SG_DEBUG("Entering\n");

	REQUIRE(has_features(), "No features assigned to kernel\n")
	REQUIRE(lhs_equals_rhs, "The kernel matrix is not symmetric!\n")
	REQUIRE(block_begin>=0 && block_begin<num_rhs,
			"Invalid block begin index (%d, %d)!\n", block_begin, block_begin)
	REQUIRE(block_begin+block_size<=num_rhs,
			"Invalid block size (%d) at starting index (%d, %d)! "
			"Please use smaller blocks!", block_size, block_begin, block_begin)
	REQUIRE(block_size>=1, "Invalid block size (%d)!\n", block_size)

	// initialize the matrix that accumulates the row/col-wise sum on the go
	// the first column stores the sum of kernel values
	// the second column stores the sum of squared kernel values
	SGMatrix<float64_t> row_sum(block_size, 2);
	row_sum.set_const(0.0);

	// since the block is symmetric with main diagonal inside, we can save half
	// the computation with using only the upper triangular part
	// this can be done in parallel for the rows/cols
#pragma omp parallel for
	for (index_t i=0; i<block_size; ++i)
	{
		// compute the kernel values on the upper triangular part of the kernel
		// matrix and compute row-wise sum and squared sum on the fly
		for (index_t j=i+1; j<block_size; ++j)
		{
			float64_t k=kernel(i+block_begin, j+block_begin);
#pragma omp critical
			{
				row_sum(i, 0)+=k;
				row_sum(j, 0)+=k;
				row_sum(i, 1)+=k*k;
				row_sum(j, 1)+=k*k;
			}
		}
	}

	// add the diagonal elements if required - keeping this check
	// outside of the loop to save cycles
	if (!no_diag)
	{
#pragma omp parallel for
		for (index_t i=0; i<block_size; ++i)
		{
			float64_t diag=kernel(i+block_begin, i+block_begin);
			row_sum(i, 0)+=diag;
			row_sum(i, 1)+=diag*diag;
		}
	}

	SG_DEBUG("Leaving\n");

	return row_sum;
}

SGVector<float64_t> CKernel::row_col_wise_sum_block(index_t block_begin_row,
		index_t block_begin_col, index_t block_size_row,
		index_t block_size_col, bool no_diag)
{
	SG_DEBUG("Entering\n");

	REQUIRE(has_features(), "No features assigned to kernel\n")
	REQUIRE(block_begin_row>=0 && block_begin_row<num_lhs &&
			block_begin_col>=0 && block_begin_col<num_rhs,
			"Invalid block begin index (%d, %d)!\n",
			block_begin_row, block_begin_col)
	REQUIRE(block_begin_row+block_size_row<=num_lhs &&
			block_begin_col+block_size_col<=num_rhs,
			"Invalid block size (%d, %d) at starting index (%d, %d)! "
			"Please use smaller blocks!", block_size_row, block_size_col,
			block_begin_row, block_begin_col)
	REQUIRE(block_size_row>=1 && block_size_col>=1,
			"Invalid block size (%d, %d)!\n", block_size_row, block_size_col)

	// check if removal of diagonal is required/valid
	if (no_diag && block_size_row!=block_size_col)
	{
		SG_WARNING("Not removing the main diagonal since block is not square!\n");
		no_diag=false;
	}

	// initialize the vector that accumulates the row/col-wise sum on the go
	// the first block_size_row entries store the row-wise sum of kernel values
	// the nextt block_size_col entries store the col-wise sum of kernel values
	SGVector<float64_t> sum(block_size_row+block_size_col);
	sum.set_const(0.0);

	// this can be done in parallel for the rows/cols
#pragma omp parallel for
	for (index_t i=0; i<block_size_row; ++i)
	{
		// compute the kernel values and compute sum on the fly
		for (index_t j=0; j<block_size_col; ++j)
		{
			float64_t k=no_diag && i==j ? 0 :
				kernel(i+block_begin_row, j+block_begin_col);
#pragma omp critical
			{
				sum[i]+=k;
				sum[j+block_size_row]+=k;
			}
		}
	}

	SG_DEBUG("Leaving\n");

	return sum;
}

template <class T> void* CKernel::get_kernel_matrix_helper(void* p)
{
	K_THREAD_PARAM<T>* params= (K_THREAD_PARAM<T>*) p;
	int32_t i_start=params->start;
	int32_t i_end=params->end;
	CKernel* k=params->kernel;
	T* result=params->result;
	bool symmetric=params->symmetric;
	int32_t n=params->n;
	int32_t m=params->m;
	bool verbose=params->verbose;
	int64_t total_start=params->total_start;
	int64_t total_end=params->total_end;
	int64_t total=total_start;

	for (int32_t i=i_start; i<i_end; i++)
	{
		int32_t j_start=0;

		if (symmetric)
			j_start=i;

		for (int32_t j=j_start; j<n; j++)
		{
			float64_t v=k->kernel(i,j);
			result[i+j*m]=v;

			if (symmetric && i!=j)
				result[j+i*m]=v;

			if (verbose)
			{
				total++;

				if (symmetric && i!=j)
					total++;

				if (total%100 == 0)
					SG_OBJ_PROGRESS(k, total, total_start, total_end)

				if (CSignal::cancel_computations())
					break;
			}
		}

	}

	return NULL;
}

template <class T>
SGMatrix<T> CKernel::get_kernel_matrix()
{
	T* result = NULL;

	REQUIRE(has_features(), "no features assigned to kernel\n")

	int32_t m=get_num_vec_lhs();
	int32_t n=get_num_vec_rhs();

	int64_t total_num = int64_t(m)*n;

	// if lhs == rhs and sizes match assume k(i,j)=k(j,i)
	bool symmetric= (lhs && lhs==rhs && m==n);

	SG_DEBUG("returning kernel matrix of size %dx%d\n", m, n)

	result=SG_MALLOC(T, total_num);

	int32_t num_threads=parallel->get_num_threads();
	if (num_threads < 2)
	{
		K_THREAD_PARAM<T> params;
		params.kernel=this;
		params.result=result;
		params.start=0;
		params.end=m;
		params.total_start=0;
		params.total_end=total_num;
		params.n=n;
		params.m=m;
		params.symmetric=symmetric;
		params.verbose=true;
		get_kernel_matrix_helper<T>((void*) &params);
	}
	else
	{
		pthread_t* threads = SG_MALLOC(pthread_t, num_threads-1);
		K_THREAD_PARAM<T>* params = SG_MALLOC(K_THREAD_PARAM<T>, num_threads);
		int64_t step= total_num/num_threads;

		int32_t t;

		num_threads--;
		for (t=0; t<num_threads; t++)
		{
			params[t].kernel = this;
			params[t].result = result;
			params[t].start = compute_row_start(t*step, n, symmetric);
			params[t].end = compute_row_start((t+1)*step, n, symmetric);
			params[t].total_start=t*step;
			params[t].total_end=(t+1)*step;
			params[t].n=n;
			params[t].m=m;
			params[t].symmetric=symmetric;
			params[t].verbose=false;

			int code=pthread_create(&threads[t], NULL,
					CKernel::get_kernel_matrix_helper<T>, (void*)&params[t]);

			if (code != 0)
			{
				SG_WARNING("Thread creation failed (thread %d of %d) "
						"with error:'%s'\n",t, num_threads, strerror(code));
				num_threads=t;
				break;
			}
		}

		params[t].kernel = this;
		params[t].result = result;
		params[t].start = compute_row_start(t*step, n, symmetric);
		params[t].end = m;
		params[t].total_start=t*step;
		params[t].total_end=total_num;
		params[t].n=n;
		params[t].m=m;
		params[t].symmetric=symmetric;
		params[t].verbose=true;
		get_kernel_matrix_helper<T>(&params[t]);

		for (t=0; t<num_threads; t++)
		{
			if (pthread_join(threads[t], NULL) != 0)
				SG_WARNING("pthread_join of thread %d/%d failed\n", t, num_threads)
		}

		SG_FREE(params);
		SG_FREE(threads);
	}

	SG_DONE()

	return SGMatrix<T>(result,m,n,true);
}


template SGMatrix<float64_t> CKernel::get_kernel_matrix<float64_t>();
template SGMatrix<float32_t> CKernel::get_kernel_matrix<float32_t>();

template void* CKernel::get_kernel_matrix_helper<float64_t>(void* p);
template void* CKernel::get_kernel_matrix_helper<float32_t>(void* p);

