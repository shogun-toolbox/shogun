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

#include "lib/config.h"
#include "lib/common.h"
#include "lib/io.h"
#include "lib/File.h"
#include "lib/Time.h"
#include "lib/Signal.h"

#include "base/Parallel.h"

#include "kernel/Kernel.h"
#include "kernel/IdentityKernelNormalizer.h"
#include "features/Features.h"

#include "classifier/svm/SVM.h"

#include <string.h>
#include <unistd.h>
#include <math.h>

#ifndef WIN32
#include <pthread.h>
#endif

using namespace shogun;

CKernel::CKernel(int32_t size)
: CSGObject(), kernel_matrix(NULL), lhs(NULL),
	rhs(NULL), combined_kernel_weight(1), optimization_initialized(false),
	opt_type(FASTBUTMEMHUNGRY), properties(KP_NONE), normalizer(NULL)
{
	if (size<10)
		size=10;

	cache_size=size;
#ifdef USE_SVMLIGHT
	memset(&kernel_cache, 0x0, sizeof(KERNEL_CACHE));
#endif //USE_SVMLIGHT

	if (get_is_initialized())
		SG_ERROR( "COptimizableKernel still initialized on destruction");

	set_normalizer(new CIdentityKernelNormalizer());
}


CKernel::CKernel(CFeatures* p_lhs, CFeatures* p_rhs, int32_t size) : CSGObject(),
	kernel_matrix(NULL), lhs(NULL), rhs(NULL), combined_kernel_weight(1),
	optimization_initialized(false), opt_type(FASTBUTMEMHUNGRY),
	properties(KP_NONE), normalizer(NULL)
{
	if (size<10)
		size=10;

	cache_size=size;
#ifdef USE_SVMLIGHT
	memset(&kernel_cache, 0x0, sizeof(KERNEL_CACHE));
#endif //USE_SVMLIGHT
	if (get_is_initialized())
		SG_ERROR("Kernel initialized on construction.\n");

	set_normalizer(new CIdentityKernelNormalizer());
	init(p_lhs, p_rhs);
}

CKernel::~CKernel()
{
	if (get_is_initialized())
		SG_ERROR("Kernel still initialized on destruction.\n");

	remove_lhs_and_rhs();
	SG_UNREF(normalizer);

	SG_INFO("Kernel deleted (%p).\n", this);
}

void CKernel::get_kernel_matrix(float64_t** dst, int32_t* m, int32_t* n)
{
	ASSERT(dst && m && n);

	float64_t* result = NULL;

	if (has_features())
	{
		int32_t num_vec1=get_num_vec_lhs();
		int32_t num_vec2=get_num_vec_rhs();
		*m=num_vec1;
		*n=num_vec2;

		int64_t total_num = ((int64_t) num_vec1) * num_vec2;
		int32_t num_done = 0;
		SG_DEBUG( "returning kernel matrix of size %dx%d\n", num_vec1, num_vec2);

		result=(float64_t*) malloc(sizeof(float64_t)*total_num);
		ASSERT(result);

		CSignal::clear_cancel();

		if ( lhs && lhs==rhs && num_vec1==num_vec2 )
		{
			for (int32_t i=0; i<num_vec1 && (!CSignal::cancel_computations()); i++)
			{
				for (int32_t j=i; j<num_vec1; j++)
				{
					float64_t v=kernel(i,j);

					result[i+j*num_vec1]=v;
					result[j+i*num_vec1]=v;

					if (num_done%100000)
						SG_PROGRESS(num_done, 0, total_num-1);

					if (i!=j)
						num_done+=2;
					else
						num_done+=1;
				}
			}
		}
		else
		{
			for (int32_t i=0; i<num_vec1 && (!CSignal::cancel_computations()); i++)
			{
				for (int32_t j=0; j<num_vec2; j++)
				{
					result[i+j*num_vec1]=kernel(i,j) ;

					if (num_done%100000)
						SG_PROGRESS(num_done, 0, total_num-1);

					num_done++;
				}
			}
		}

		SG_DONE();
	}
	else
      SG_ERROR( "no features assigned to kernel\n");

	*dst=result;
}

float32_t* CKernel::get_kernel_matrix_shortreal(
	int32_t &num_vec1, int32_t &num_vec2, float32_t* target)
{
	float32_t* result = NULL;

	if (has_features())
	{
		if (target && (num_vec1!=get_num_vec_lhs() ||
					num_vec2!=get_num_vec_rhs()) )
			SG_ERROR( "kernel matrix size mismatch\n");

		num_vec1=get_num_vec_lhs();
		num_vec2=get_num_vec_rhs();

		int64_t total_num = ((int64_t) num_vec1) * num_vec2;
		int32_t num_done = 0;

		SG_DEBUG( "returning kernel matrix of size %dx%d\n", num_vec1, num_vec2);

		if (target)
			result=target;
		else
			result=new float32_t[total_num];

		if (lhs && lhs==rhs && num_vec1==num_vec2)
		{
			for (int32_t i=0; i<num_vec1; i++)
			{
				for (int32_t j=i; j<num_vec1; j++)
				{
					float64_t v=kernel(i,j);

					result[i+j*num_vec1]=v;
					result[j+i*num_vec1]=v;

					if (num_done%100000)
						SG_PROGRESS(num_done, 0, total_num-1);

					if (i!=j)
						num_done+=2;
					else
						num_done+=1;
				}
			}
		}
		else
		{
			for (int32_t i=0; i<num_vec1; i++)
			{
				for (int32_t j=0; j<num_vec2; j++)
				{
					result[i+j*num_vec1]=kernel(i,j) ;

					if (num_done%100000)
						SG_PROGRESS(num_done, 0, total_num-1);

					num_done++;
				}
			}
		}

		SG_DONE();
	}
	else
      SG_ERROR( "no features assigned to kernel\n");

	return result;
}

float64_t* CKernel::get_kernel_matrix_real(
	int32_t &num_vec1, int32_t &num_vec2, float64_t* target)
{
	float64_t* result = NULL;

	if (has_features())
	{
		if (target && (num_vec1!=get_num_vec_lhs() ||
					num_vec2!=get_num_vec_rhs()) )
			SG_ERROR( "kernel matrix size mismatch\n");

		num_vec1=get_num_vec_lhs();
		num_vec2=get_num_vec_rhs();

		int64_t total_num = ((int64_t) num_vec1) * num_vec2;
		int32_t num_done = 0;

		SG_DEBUG( "returning kernel matrix of size %dx%d\n", num_vec1, num_vec2);

		if (target)
			result=target;
		else
			result=new float64_t[total_num];

		if (lhs && lhs==rhs && num_vec1==num_vec2)
		{
			for (int32_t i=0; i<num_vec1; i++)
			{
				for (int32_t j=i; j<num_vec1; j++)
				{
					float64_t v=kernel(i,j);

					result[i+j*num_vec1]=v;
					result[j+i*num_vec1]=v;

					if (num_done%100000)
						SG_PROGRESS(num_done, 0, total_num-1);

					if (i!=j)
						num_done+=2;
					else
						num_done+=1;
				}
			}
		}
		else
		{
			for (int32_t i=0; i<num_vec1; i++)
			{
				for (int32_t j=0; j<num_vec2; j++)
				{
					result[i+j*num_vec1]=kernel(i,j) ;

					if (num_done%100000)
						SG_PROGRESS(num_done, 0, total_num-1);

					num_done++;
				}
			}
		}

		SG_DONE();
	}
	else
      SG_ERROR( "no features assigned to kernel\n");

	return result;
}


#ifdef USE_SVMLIGHT
void CKernel::resize_kernel_cache(KERNELCACHE_IDX size, bool regression_hack)
{
	if (size<10)
		size=10;
	
	kernel_cache_cleanup();
	cache_size=size;

	if (has_features())
		kernel_cache_init(cache_size, regression_hack);
}
#endif //USE_SVMLIGHT

bool CKernel::init(CFeatures* l, CFeatures* r)
{
	//make sure features were indeed supplied
	ASSERT(l);
	ASSERT(r);

	//make sure features are compatible
	ASSERT(l->get_feature_class()==r->get_feature_class());
	ASSERT(l->get_feature_type()==r->get_feature_type());

	//remove references to previous features
	remove_lhs_and_rhs();

    //increase reference counts
    SG_REF(l);
    if (l!=r)
        SG_REF(r);

	lhs=l;
	rhs=r;

	return true;
}

bool CKernel::set_normalizer(CKernelNormalizer* n)
{
	SG_REF(n);
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
	ASSERT(totdoc>0);
	uint64_t buffer_size=0;
	int32_t i;

	//in regression the additional constraints are made by doubling the training data
	if (regression_hack)
		totdoc*=2;

	buffer_size=((uint64_t) buffsize)*1024*1024/sizeof(KERNELCACHE_ELEM);
	if (buffer_size>((uint64_t) totdoc)*totdoc)
		buffer_size=((uint64_t) totdoc)*totdoc;

	SG_INFO( "using a kernel cache of size %lld MB (%lld bytes) for %s Kernel\n", buffer_size*sizeof(KERNELCACHE_ELEM)/1024/1024, buffer_size*sizeof(KERNELCACHE_ELEM), get_name());

	//make sure it fits in the *signed* KERNELCACHE_IDX type
	ASSERT(buffer_size < (((uint64_t) 1) << (sizeof(KERNELCACHE_IDX)*8-1)));

	kernel_cache.index = new int32_t[totdoc];
	kernel_cache.occu = new int32_t[totdoc];
	kernel_cache.lru = new int32_t[totdoc];
	kernel_cache.invindex = new int32_t[totdoc];
	kernel_cache.active2totdoc = new int32_t[totdoc];
	kernel_cache.totdoc2active = new int32_t[totdoc];
	kernel_cache.buffer = new KERNELCACHE_ELEM[buffer_size];
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
					buffer[j]=(float64_t) kernel(docnum, j);
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
				buffer[j]=(KERNELCACHE_ELEM) kernel(docnum, j);
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
					cache[j]=kernel(m, k);
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
				cache[j]=params->kernel->kernel(m, k);
		}

		//now line m is cached
		params->needs_computation[m]=0;
	}
	return NULL;
}

// Fills cache for the rows in key 
void CKernel::cache_multiple_kernel_rows(int32_t* rows, int32_t num_rows)
{
#ifndef WIN32
	if (parallel->get_num_threads()<2)
	{
#endif
		for(int32_t i=0;i<num_rows;i++) 
			cache_kernel_row(rows[i]);
#ifndef WIN32
	}
	else
	{
		// fill up kernel cache 
		int32_t* uncached_rows = new int32_t[num_rows];
		KERNELCACHE_ELEM** cache = new KERNELCACHE_ELEM*[num_rows];
		pthread_t* threads = new pthread_t[parallel->get_num_threads()-1];
		S_KTHREAD_PARAM* params = new S_KTHREAD_PARAM[parallel->get_num_threads()-1];
		int32_t num_threads=parallel->get_num_threads()-1;
		int32_t num_vec=get_num_vec_lhs();
		ASSERT(num_vec>0);
		uint8_t* needs_computation=new uint8_t[num_vec];
		memset(needs_computation, 0, sizeof(uint8_t)*num_vec);
		int32_t step=0;
		int32_t num=0;
		int32_t end=0;

		// allocate cachelines if necessary
		for (int32_t i=0; i<num_rows; i++)
		{
			int32_t idx=rows[i];
			if (kernel_cache_check(idx))
				continue;

			if (idx>=num_vec)
				idx=2*num_vec-1-idx;

			needs_computation[idx]=1;
			uncached_rows[num]=idx;
			cache[num]= kernel_cache_clean_and_malloc(idx);

			if (!cache[num])
				SG_ERROR("Kernel cache full! => increase cache size\n");

			num++;
		}

		if (num>0)
		{
			step= num/parallel->get_num_threads();

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
				end=params[t].end;

				if (pthread_create(&threads[t], NULL, CKernel::cache_multiple_kernel_row_helper, (void*)&params[t]) != 0)
				{
					num_threads=t;
					end=t*step;
					SG_WARNING("thread creation failed\n");
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

		cache_multiple_kernel_row_helper(&last_param);


		for (int32_t t=0; t<num_threads; t++)
		{
			if (pthread_join(threads[t], NULL) != 0)
				SG_WARNING( "pthread_join failed\n");
		}

		delete[] needs_computation;
		delete[] params;
		delete[] threads;
		delete[] cache;
		delete[] uncached_rows;
	}
#endif
}

// remove numshrink columns in the cache
// which correspond to examples marked
void CKernel::kernel_cache_shrink(
	int32_t totdoc, int32_t numshrink, int32_t *after)
{                           
	register int32_t i,j,jj,scount;     // 0 in after.
	KERNELCACHE_IDX from=0,to=0;
	int32_t *keep;

	keep=new int32_t[totdoc];
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

	kernel_cache.max_elems=
		(int32_t)(kernel_cache.buffsize/kernel_cache.activenum);
	if(kernel_cache.max_elems>totdoc) {
		kernel_cache.max_elems=totdoc;
	}

	delete[] keep;

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
	delete[] kernel_cache.index;
	delete[] kernel_cache.occu;
	delete[] kernel_cache.lru;
	delete[] kernel_cache.invindex;
	delete[] kernel_cache.active2totdoc;
	delete[] kernel_cache.totdoc2active;
	delete[] kernel_cache.buffer;
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

bool CKernel::load(char* fname)
{
	return false;
}

bool CKernel::save(char* fname)
{
	int32_t i=0;
	int32_t num_left=get_num_vec_lhs();
	int32_t num_right=rhs->get_num_vectors();
	KERNELCACHE_IDX num_total=num_left*num_right;

	CFile f(fname, 'w', F_DREAL);

    for (int32_t l=0; l< (int32_t) num_left && f.is_ok(); l++)
	{
		for (int32_t r=0; r< (int32_t) num_right && f.is_ok(); r++)
		{
			if (!(i % (num_total/10+1)))
				SG_PRINT("%02d%%.", (int32_t) (100.0*i/num_total));
			else if (!(i % (num_total/200+1)))
				SG_PRINT(".");

			float64_t k=kernel(l,r);
			f.save_real_data(&k, 1);

			i++;
		}
	}

	if (f.is_ok())
		SG_INFO( "kernel matrix of size %ld x %ld written (filesize: %ld)\n", num_left, num_right, num_total*sizeof(KERNELCACHE_ELEM));

    return (f.is_ok());
}

void CKernel::remove_lhs_and_rhs()
{
	if (rhs!=lhs)
		SG_UNREF(rhs);
	rhs = NULL;

	SG_UNREF(lhs);
	lhs = NULL;

#ifdef USE_SVMLIGHT
	cache_reset();
#endif //USE_SVMLIGHT
}

void CKernel::remove_lhs()
{ 
	if (rhs==lhs)
		rhs=NULL;
	SG_UNREF(lhs);
	lhs = NULL;

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

#ifdef USE_SVMLIGHT
	cache_reset();
#endif //USE_SVMLIGHT
}


void CKernel::list_kernel()
{
	SG_INFO( "%p - \"%s\" weight=%1.2f OPT:%s", this, get_name(),
			get_combined_kernel_weight(),
			get_optimization_type()==FASTBUTMEMHUNGRY ? "FASTBUTMEMHUNGRY" :
			"SLOWBUTMEMEFFICIENT");

	switch (get_kernel_type())
	{
		case K_UNKNOWN:
			SG_INFO( "K_UNKNOWN ");
			break;
		case K_LINEAR:
			SG_INFO( "K_LINEAR ");
			break;
		case K_SPARSELINEAR:
			SG_INFO( "K_SPARSELINEAR ");
			break;
		case K_POLY:
			SG_INFO( "K_POLY ");
			break;
		case K_GAUSSIAN:
			SG_INFO( "K_GAUSSIAN ");
			break;
		case K_SPARSEGAUSSIAN:
			SG_INFO( "K_SPARSEGAUSSIAN ");
			break;
		case K_GAUSSIANSHIFT:
			SG_INFO( "K_GAUSSIANSHIFT ");
			break;
		case K_HISTOGRAM:
			SG_INFO( "K_HISTOGRAM ");
			break;
		case K_SALZBERG:
			SG_INFO( "K_SALZBERG ");
			break;
		case K_LOCALITYIMPROVED:
			SG_INFO( "K_LOCALITYIMPROVED ");
			break;
		case K_SIMPLELOCALITYIMPROVED:
			SG_INFO( "K_SIMPLELOCALITYIMPROVED ");
			break;
		case K_FIXEDDEGREE:
			SG_INFO( "K_FIXEDDEGREE ");
			break;
		case K_WEIGHTEDDEGREE:
			SG_INFO( "K_WEIGHTEDDEGREE ");
			break;
		case K_WEIGHTEDDEGREEPOS:
			SG_INFO( "K_WEIGHTEDDEGREEPOS ");
			break;
		case K_WEIGHTEDCOMMWORDSTRING:
			SG_INFO( "K_WEIGHTEDCOMMWORDSTRING ");
			break;
		case K_POLYMATCH:
			SG_INFO( "K_POLYMATCH ");
			break;
		case K_ALIGNMENT:
			SG_INFO( "K_ALIGNMENT ");
			break;
		case K_COMMWORDSTRING:
			SG_INFO( "K_COMMWORDSTRING ");
			break;
		case K_COMMULONGSTRING:
			SG_INFO( "K_COMMULONGSTRING ");
			break;
		case K_COMBINED:
			SG_INFO( "K_COMBINED ");
			break;
		case K_AUC:
			SG_INFO( "K_AUC ");
			break;
		case K_CUSTOM:
			SG_INFO( "K_CUSTOM ");
			break;
		case K_SIGMOID:
			SG_INFO( "K_SIGMOID ");
			break;
		case K_CHI2:
			SG_INFO( "K_CHI2 ");
			break;
		case K_DIAG:
			SG_INFO( "K_DIAG ");
			break;
		case K_CONST:
			SG_INFO( "K_CONST ");
			break;
		case K_DISTANCE:
			SG_INFO( "K_DISTANCE ");
			break;
		case K_LOCALALIGNMENT:
			SG_INFO( "K_LOCALALIGNMENT ");
			break;
		case K_TPPK:
			SG_INFO( "K_TPPK ");
			break;
		default:
         SG_ERROR( "ERROR UNKNOWN KERNEL TYPE");
			break;
	}

	switch (get_feature_class())
	{
		case C_UNKNOWN:
			SG_INFO( "C_UNKNOWN ");
			break;
		case C_SIMPLE:
			SG_INFO( "C_SIMPLE ");
			break;
		case C_SPARSE:
			SG_INFO( "C_SPARSE ");
			break;
		case C_STRING:
			SG_INFO( "C_STRING ");
			break;
		case C_COMBINED:
			SG_INFO( "C_COMBINED ");
			break;
		case C_ANY:
			SG_INFO( "C_ANY ");
			break;
		default:
         SG_ERROR( "ERROR UNKNOWN FEATURE CLASS");
	}

	switch (get_feature_type())
	{
		case F_UNKNOWN:
			SG_INFO( "F_UNKNOWN ");
			break;
		case F_DREAL:
			SG_INFO( "F_REAL ");
			break;
		case F_SHORT:
			SG_INFO( "F_SHORT ");
			break;
		case F_CHAR:
			SG_INFO( "F_CHAR ");
			break;
		case F_INT:
			SG_INFO( "F_INT ");
			break;
		case F_BYTE:
			SG_INFO( "F_BYTE ");
			break;
		case F_WORD:
			SG_INFO( "F_WORD ");
			break;
		case F_ULONG:
			SG_INFO( "F_ULONG ");
			break;
		case F_ANY:
			SG_INFO( "F_ANY ");
			break;
		default:
         SG_ERROR( "ERROR UNKNOWN FEATURE TYPE");
			break;
	}
	SG_INFO( "\n");
}

bool CKernel::init_optimization(
	int32_t count, int32_t *IDX, float64_t * weights)
{
   SG_ERROR( "kernel does not support linadd optimization\n");
	return false ;
}

bool CKernel::delete_optimization() 
{
   SG_ERROR( "kernel does not support linadd optimization\n");
	return false;
}

float64_t CKernel::compute_optimized(int32_t vector_idx)
{
   SG_ERROR( "kernel does not support linadd optimization\n");
	return 0;
}

void CKernel::compute_batch(
	int32_t num_vec, int32_t* vec_idx, float64_t* target, int32_t num_suppvec,
	int32_t* IDX, float64_t* weights, float64_t factor)
{
   SG_ERROR( "kernel does not support batch computation\n");
}

void CKernel::add_to_normal(int32_t vector_idx, float64_t weight)
{
   SG_ERROR( "kernel does not support linadd optimization, add_to_normal not implemented\n");
}

void CKernel::clear_normal()
{
   SG_ERROR( "kernel does not support linadd optimization, clear_normal not implemented\n");
}

int32_t CKernel::get_num_subkernels()
{
	return 1;
}

void CKernel::compute_by_subkernel(
	int32_t vector_idx, float64_t * subkernel_contrib)
{
   SG_ERROR( "kernel compute_by_subkernel not implemented\n");
}

const float64_t* CKernel::get_subkernel_weights(int32_t &num_weights)
{
	num_weights=1 ;
	return &combined_kernel_weight ;
}

void CKernel::set_subkernel_weights(float64_t* weights, int32_t num_weights)
{
	combined_kernel_weight = weights[0] ;
	if (num_weights!=1)
      SG_ERROR( "number of subkernel weights should be one ...\n");
}

bool CKernel::init_optimization_svm(CSVM * svm)
{
	int32_t num_suppvec=svm->get_num_support_vectors();
	int32_t* sv_idx=new int32_t[num_suppvec];
	float64_t* sv_weight=new float64_t[num_suppvec];

	for (int32_t i=0; i<num_suppvec; i++)
	{
		sv_idx[i]    = svm->get_support_vector(i);
		sv_weight[i] = svm->get_alpha(i);
	}
	bool ret = init_optimization(num_suppvec, sv_idx, sv_weight);

	delete[] sv_idx;
	delete[] sv_weight;
	return ret;
}

