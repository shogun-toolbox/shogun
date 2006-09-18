/*
 * EXCEPT FOR THE KERNEL CACHING FUNCTIONS WHICH ARE (W) THORSTEN JOACHIMS
 * COPYRIGHT (C) 1999  UNIVERSITAET DORTMUND - ALL RIGHTS RESERVED
 *
 * this program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Written (W) 1999-2006 Gunnar Raetsch
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/config.h"

#include "lib/common.h"
#include "lib/io.h"
#include "lib/File.h"
#include "lib/Time.h"
#include "lib/Parallel.h"

#include "kernel/Kernel.h"
#include "features/Features.h"

#include <string.h>
#include <unistd.h>
#include <pthread.h>

CKernel::CKernel(INT size) 
: kernel_matrix(NULL), precomputed_matrix(NULL),
	precompute_subkernel_matrix(false), precompute_matrix(false), 
	lhs(NULL), rhs(NULL), combined_kernel_weight(1), optimization_initialized(false),
	opt_type(FASTBUTMEMHUNGRY), properties(KP_NONE)
{
	if (size<10)
		size=10;

	cache_size=size;
#ifdef USE_SVMLIGHT
	memset(&kernel_cache, 0x0, sizeof(KERNEL_CACHE));
#endif
	if (get_is_initialized()) 
		CIO::message(M_ERROR, "COptimizableKernel still initialized on destruction") ;
}

		
CKernel::CKernel(CFeatures* lhs, CFeatures* rhs, INT size)
: kernel_matrix(NULL), precomputed_matrix(NULL),
	precompute_subkernel_matrix(false), precompute_matrix(false), 
	lhs(NULL), rhs(NULL), combined_kernel_weight(1), optimization_initialized(false),
	opt_type(FASTBUTMEMHUNGRY), properties(KP_NONE)
{
	if (size<10)
		size=10;

	cache_size=size;
#ifdef USE_SVMLIGHT
	memset(&kernel_cache, 0x0, sizeof(KERNEL_CACHE));
#endif
	if (get_is_initialized()) 
		CIO::message(M_ERROR, "COptimizableKernel still initialized on destruction") ;

	init(lhs, rhs, true);
}

CKernel::~CKernel()
{
	if (get_is_initialized()) 
		CIO::message(M_ERROR, "COptimizableKernel still initialized on destruction") ;

#ifdef USE_SVMLIGHT
	kernel_cache_cleanup();
#endif

	delete[] precomputed_matrix ;
	precomputed_matrix=NULL ;
}

SHORTREAL* CKernel::get_kernel_matrix_shortreal(int &num_vec1, int &num_vec2, SHORTREAL* target)
{
	SHORTREAL* result = NULL;
	CFeatures* f1 = get_lhs();
	CFeatures* f2 = get_rhs();

	if (f1 && f2)
	{
		if (target && (num_vec1!=f1->get_num_vectors() || num_vec2!=f2->get_num_vectors()) )
			CIO::message(M_ERROR, "kernel matrix does not fit into target\n");

		num_vec1=f1->get_num_vectors();
		num_vec2=f2->get_num_vectors();
		int total_num = num_vec1 * num_vec2;
		int num_done = 0;

		CIO::message(M_DEBUG, "returning kernel matrix of size %dx%d\n", num_vec1, num_vec2);

		if (target)
			result=target;
		else
			result=new SHORTREAL[total_num];

		ASSERT(result);

		if ( (f1 == f2) && (num_vec1 == num_vec2) )
		{
			for (int i=0; i<num_vec1; i++)
			{
				for (int j=i; j<num_vec1; j++)
				{
					double v=kernel(i,j);

					result[i+j*num_vec1]=v;
					result[j+i*num_vec1]=v;

					if (num_done%100000)
						CIO::progress(num_done, 0, total_num-1);

					if (i!=j)
						num_done+=2;
					else
						num_done+=1;
				}
			}
		}
		else
		{
			for (int i=0; i<num_vec1; i++)
			{
				for (int j=0; j<num_vec2; j++)
				{
					result[i+j*num_vec1]=kernel(i,j) ;

					if (num_done%100000)
						CIO::progress(num_done, 0, total_num-1);

					num_done++;
				}
			}
		}

		CIO::message(M_MESSAGEONLY, "done.           \n");
	}
	else
		CIO::message(M_ERROR, "no features assigned to kernel\n");

	return result;
}

DREAL* CKernel::get_kernel_matrix_real(int &num_vec1, int &num_vec2, DREAL* target)
{
	DREAL* result = NULL;
	CFeatures* f1 = get_lhs();
	CFeatures* f2 = get_rhs();

	if (f1 && f2)
	{
		if (target && (num_vec1!=f1->get_num_vectors() || num_vec2!=f2->get_num_vectors()) )
			CIO::message(M_ERROR, "kernel matrix does not fit into target\n");

		num_vec1=f1->get_num_vectors();
		num_vec2=f2->get_num_vectors();
		int total_num = num_vec1 * num_vec2;
		int num_done = 0;

		CIO::message(M_DEBUG, "returning kernel matrix of size %dx%d\n", num_vec1, num_vec2);

		if (target)
			result=target;
		else
			result=new DREAL[total_num];

		ASSERT(result);

		if ( (f1 == f2) && (num_vec1 == num_vec2) )
		{
			for (int i=0; i<num_vec1; i++)
			{
				for (int j=i; j<num_vec1; j++)
				{
					double v=kernel(i,j);

					result[i+j*num_vec1]=v;
					result[j+i*num_vec1]=v;

					if (num_done%100000)
						CIO::progress(num_done, 0, total_num-1);

					if (i!=j)
						num_done+=2;
					else
						num_done+=1;
				}
			}
		}
		else
		{
			for (int i=0; i<num_vec1; i++)
			{
				for (int j=0; j<num_vec2; j++)
				{
					result[i+j*num_vec1]=kernel(i,j) ;

					if (num_done%100000)
						CIO::progress(num_done, 0, total_num-1);

					num_done++;
				}
			}
		}

		CIO::message(M_MESSAGEONLY, "done.           \n");
	}
	else
		CIO::message(M_ERROR, "no features assigned to kernel\n");

	return result;
}


#ifdef USE_SVMLIGHT
void CKernel::resize_kernel_cache(KERNELCACHE_IDX size, bool regression_hack)
{
	if (size<10)
		size=10;
	
	kernel_cache_cleanup();
	cache_size=size;

	memset(&kernel_cache, 0x0, sizeof(KERNEL_CACHE));

	if (lhs!=NULL && rhs!=NULL)
		kernel_cache_init(cache_size, regression_hack);
}
#endif

bool CKernel::init(CFeatures* l, CFeatures* r, bool do_init)
{
	//make sure features were indeed supplied
	ASSERT(l);
	ASSERT(r);

	//make sure features are compatible
	ASSERT(l->get_feature_class() == r->get_feature_class());
	ASSERT(l->get_feature_type() == r->get_feature_type());

	lhs=l;
	rhs=r;

#ifdef USE_SVMLIGHT
	// allocate kernel cache but clean up beforehand
	kernel_cache_cleanup();
	kernel_cache_init(cache_size);
#endif

	delete[] precomputed_matrix ;
	precomputed_matrix=NULL ;

	return true;
}

#ifdef USE_SVMLIGHT
/****************************** Cache handling *******************************/

void CKernel::kernel_cache_init(INT buffsize, bool regression_hack)
{
	INT i;
	INT totdoc=get_lhs()->get_num_vectors();
	ULONG buffer_size=0;

	//in regression the additional constraints are made by doubling the training data
	if (regression_hack)
		totdoc*=2;

	buffer_size=((ULONG) buffsize)*1024*1024/sizeof(KERNELCACHE_ELEM);
	if (buffer_size>((ULONG) totdoc)*totdoc)
		buffer_size=((ULONG) totdoc)*totdoc;

	CIO::message(M_INFO, "using a kernel cache of size %i MB (%lld bytes) for %s Kernel\n", buffer_size*sizeof(KERNELCACHE_ELEM)/1024/1024, buffer_size*sizeof(KERNELCACHE_ELEM), get_name());

	//make sure it fits in the *signed* KERNELCACHE_IDX type
	ASSERT(buffer_size < (((ULONG) 1) << (sizeof(KERNELCACHE_IDX)*8-1)));

	kernel_cache.index = new INT[totdoc];
	kernel_cache.occu = new INT[totdoc];
	kernel_cache.lru = new INT[totdoc];
	kernel_cache.invindex = new INT[totdoc];
	kernel_cache.active2totdoc = new INT[totdoc];
	kernel_cache.totdoc2active = new INT[totdoc];
	kernel_cache.buffer = new KERNELCACHE_ELEM[buffer_size];
	kernel_cache.buffsize=buffer_size;
	kernel_cache.max_elems=(INT) (kernel_cache.buffsize/totdoc);

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

void CKernel::get_kernel_row(INT docnum, INT *active2dnum, DREAL *buffer, bool full_line)
{
	INT i,j;
	KERNELCACHE_IDX start;

	int num_vectors = lhs->get_num_vectors();
	if (docnum>=num_vectors)
		docnum=2*num_vectors-1-docnum;

	/* is cached? */
	if(kernel_cache.index[docnum] != -1) 
	{
		kernel_cache.lru[kernel_cache.index[docnum]]=kernel_cache.time; /* lru */
		start=((KERNELCACHE_IDX) kernel_cache.activenum)*kernel_cache.index[docnum];

		if (full_line)
		{
			for(j=0;j<get_lhs()->get_num_vectors();j++)
			{
				if(kernel_cache.totdoc2active[j] >= 0)
					buffer[j]=kernel_cache.buffer[start+kernel_cache.totdoc2active[j]];
				else
					buffer[j]=(DREAL) kernel(docnum, j);
			}
		}
		else
		{
			for(i=0;(j=active2dnum[i])>=0;i++)
			{
				if(kernel_cache.totdoc2active[j] >= 0)
					buffer[j]=kernel_cache.buffer[start+kernel_cache.totdoc2active[j]];
				else
					buffer[j]=(DREAL) kernel(docnum, j);
			}
		}
	}
	else 
	{
		if (full_line)
		{
			for(j=0;j<get_lhs()->get_num_vectors();j++)
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
void CKernel::cache_kernel_row(INT m)
{
	register INT j,k,l;
	register KERNELCACHE_ELEM *cache;

	int num_vectors = lhs->get_num_vectors();

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
	INT j,k,l;
	S_KTHREAD_PARAM* params = (S_KTHREAD_PARAM*) p;

	for (INT i=params->start; i<params->end; i++)
	{
		KERNELCACHE_ELEM* cache=params->cache[i];
		INT m = params->uncached_rows[i];
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
void CKernel::cache_multiple_kernel_rows(INT* rows, INT num_rows)
{
	if (CParallel::get_num_threads()<2)
	{
		for(INT i=0;i<num_rows;i++) 
			cache_kernel_row(rows[i]);
	}
	else
	{
		// fill up kernel cache 
		INT uncached_rows[num_rows];
		KERNELCACHE_ELEM* cache[num_rows];
		pthread_t threads[CParallel::get_num_threads()-1];
		S_KTHREAD_PARAM params[CParallel::get_num_threads()-1];
		INT num_threads=CParallel::get_num_threads()-1;
		INT num_vec=get_lhs()->get_num_vectors();
		ASSERT(num_vec>0);
		BYTE* needs_computation=new BYTE[num_vec];
		ASSERT(needs_computation);
		memset(needs_computation,0,sizeof(BYTE)*num_vec);
		INT step=0;
		INT num=0;
		INT end=0;

		// allocate cachelines if necessary
		for(INT i=0; i<num_rows; i++) 
		{
			if(!kernel_cache_check(rows[i])) 
			{
				needs_computation[rows[i]]=1;
				uncached_rows[num]=rows[i];
				cache[num]= kernel_cache_clean_and_malloc(rows[i]);

				if (!cache[num] )
					CIO::message(M_ERROR, "Kernel cache full! => increase cache size");

				num++;
			}
		}

		if (num>0)
		{
			step= num/CParallel::get_num_threads();

			if (step<1)
			{
				num_threads=num-1;
				step=1;
			}

			for (INT t=0; t<num_threads; t++)
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
					CIO::message(M_WARN,"thread creation failed\n");
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


		for (INT t=0; t<num_threads; t++)
		{
			if (pthread_join(threads[t], NULL) != 0)
				CIO::message(M_WARN, "pthread_join failed\n");
		}

		delete[] needs_computation;
	}
}

// remove numshrink columns in the cache
// which correspond to examples marked
void CKernel::kernel_cache_shrink(INT totdoc, INT numshrink, INT *after)
{                           
	register INT i,j,jj,scount;     // 0 in after.
	KERNELCACHE_IDX from=0,to=0;
	INT *keep;

	keep=new INT[totdoc];
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

	kernel_cache.max_elems=(INT)(kernel_cache.buffsize/kernel_cache.activenum);
	if(kernel_cache.max_elems>totdoc) {
		kernel_cache.max_elems=totdoc;
	}

	delete[] keep;

}

void CKernel::kernel_cache_reset_lru()
{
	INT maxlru=0,k;

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

INT CKernel::kernel_cache_malloc()
{
  INT i;

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

void CKernel::kernel_cache_free(INT cacheidx)
{
	kernel_cache.occu[cacheidx]=0;
	kernel_cache.elems--;
}

// remove least recently used cache
// element
INT CKernel::kernel_cache_free_lru()  
{                                     
  register INT k,least_elem=-1,least_time;

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
KERNELCACHE_ELEM* CKernel::kernel_cache_clean_and_malloc(INT cacheidx)
{             
	INT result;
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
#endif

bool CKernel::load(CHAR* fname)
{
	return false;
}

bool CKernel::save(CHAR* fname)
{
	INT i=0;
	INT num_left=get_lhs()->get_num_vectors();
	INT num_right=get_rhs()->get_num_vectors();
	KERNELCACHE_IDX num_total=num_left*num_right;

	CFile f(fname, 'w', F_DREAL);

    for (INT l=0; l< (INT) num_left && f.is_ok(); l++)
	{
		for (INT r=0; r< (INT) num_right && f.is_ok(); r++)
		{
			if (!(i % (num_total/10+1)))
				CIO::message(M_MESSAGEONLY, "%02d%%.", (int) (100.0*i/num_total));
			else if (!(i % (num_total/200+1)))
				CIO::message(M_MESSAGEONLY, ".");

			double k=kernel(l,r);
			f.save_real_data(&k, 1);

			i++;
		}
	}

	if (f.is_ok())
		CIO::message(M_INFO, "kernel matrix of size %ld x %ld written (filesize: %ld)\n", num_left, num_right, num_total*sizeof(KERNELCACHE_ELEM));

    return (f.is_ok());
}

void CKernel::remove_lhs()
{ 
#ifdef USE_SVMLIGHT
	if (lhs)
		cache_reset();
#endif

	lhs = NULL;
}

/// takes all necessary steps if the rhs is removed from kernel
void CKernel::remove_rhs()
{
#ifdef USE_SVMLIGHT
	if (rhs)
		cache_reset();
#endif
	rhs = NULL;
}


void CKernel::list_kernel()
{
	CIO::message(M_INFO, "0x%X - \"%s\" weight=%1.2f OPT:%s", this, get_name(), get_combined_kernel_weight(), get_optimization_type()==FASTBUTMEMHUNGRY ? "FASTBUTMEMHUNGRY" : "SLOWBUTMEMEFFICIENT");
	switch (get_kernel_type())
	{
		case K_UNKNOWN:
			CIO::message(M_INFO, "K_UNKNOWN ");
			break;
		case K_LINEAR:
			CIO::message(M_INFO, "K_LINEAR ");
			break;
		case K_POLY:
			CIO::message(M_INFO, "K_POLY ");
			break;
		case K_GAUSSIAN:
			CIO::message(M_INFO, "K_GAUSSIAN ");
			break;
		case K_HISTOGRAM:
			CIO::message(M_INFO, "K_HISTOGRAM ");
			break;
		case K_SALZBERG:
			CIO::message(M_INFO, "K_SALZBERG ");
			break;
		case K_LOCALITYIMPROVED:
			CIO::message(M_INFO, "K_LOCALITYIMPROVED ");
			break;
		case K_SIMPLELOCALITYIMPROVED:
			CIO::message(M_INFO, "K_SIMPLELOCALITYIMPROVED ");
			break;
		case K_FIXEDDEGREE:
			CIO::message(M_INFO, "K_FIXEDDEGREE ");
			break;
		case K_WEIGHTEDDEGREE:
			CIO::message(M_INFO, "K_WEIGHTEDDEGREE ");
			break;
		case K_WEIGHTEDDEGREEPOS:
			CIO::message(M_INFO, "K_WEIGHTEDDEGREEPOS ");
			break;
		case K_COMMWORD:
			CIO::message(M_INFO, "K_COMMWORD ");
			break;
		case K_POLYMATCH:
			CIO::message(M_INFO, "K_POLYMATCH ");
			break;
		case K_ALIGNMENT:
			CIO::message(M_INFO, "K_ALIGNMENT ");
			break;
		case K_COMMWORDSTRING:
			CIO::message(M_INFO, "K_COMMWORDSTRING ");
			break;
		case K_COMMULONGSTRING:
			CIO::message(M_INFO, "K_COMMULONGSTRING ");
			break;
		case K_SPARSENORMSQUARED:
			CIO::message(M_INFO, "K_SPARSENORMSQUARED ");
			break;
		case K_COMBINED:
			CIO::message(M_INFO, "K_COMBINED ");
			break;
		case K_CUSTOM:
			CIO::message(M_INFO, "K_CUSTOM ");
			break;
		case K_DIAG:
			CIO::message(M_INFO, "K_DIAG ");
			break;
		default:
			CIO::message(M_ERROR, "ERROR UNKNOWN KERNEL TYPE");
			break;
	}

	switch (get_feature_class())
	{
		case C_UNKNOWN:
			CIO::message(M_INFO, "C_UNKNOWN ");
			break;
		case C_SIMPLE:
			CIO::message(M_INFO, "C_SIMPLE ");
			break;
		case C_SPARSE:
			CIO::message(M_INFO, "C_SPARSE ");
			break;
		case C_STRING:
			CIO::message(M_INFO, "C_STRING ");
			break;
		case C_COMBINED:
			CIO::message(M_INFO, "C_COMBINED ");
			break;
		case C_ANY:
			CIO::message(M_INFO, "C_ANY ");
			break;
		default:
			CIO::message(M_ERROR, "ERROR UNKNOWN FEATURE CLASS");
	}

	switch (get_feature_type())
	{
		case F_UNKNOWN:
			CIO::message(M_INFO, "F_UNKNOWN ");
			break;
		case F_DREAL:
			CIO::message(M_INFO, "F_REAL ");
			break;
		case F_SHORT:
			CIO::message(M_INFO, "F_SHORT ");
			break;
		case F_CHAR:
			CIO::message(M_INFO, "F_CHAR ");
			break;
		case F_INT:
			CIO::message(M_INFO, "F_INT ");
			break;
		case F_BYTE:
			CIO::message(M_INFO, "F_BYTE ");
			break;
		case F_WORD:
			CIO::message(M_INFO, "F_WORD ");
			break;
		case F_ULONG:
			CIO::message(M_INFO, "F_ULONG ");
			break;
		case F_ANY:
			CIO::message(M_INFO, "F_ANY ");
			break;
		default:
			CIO::message(M_ERROR, "ERROR UNKNOWN FEATURE TYPE");
			break;
	}
	CIO::message(M_INFO, "\n");
}

bool CKernel::init_optimization(INT count, INT *IDX, DREAL * weights)
{
	CIO::message(M_ERROR, "kernel does not support linadd optimization\n") ;
	return false ;
}

bool CKernel::delete_optimization() 
{
	CIO::message(M_ERROR, "kernel does not support linadd optimization\n") ;
	return false;
}

DREAL CKernel::compute_optimized(INT idx)
{
	CIO::message(M_ERROR, "kernel does not support linadd optimization\n") ;
	return 0;
}

DREAL* CKernel::compute_batch(INT& num_vec, DREAL* target, INT num_suppvec, INT* IDX, DREAL* weights, DREAL factor)
{
	CIO::message(M_ERROR, "kernel does not support batch computation\n") ;
	num_vec=0;
	return NULL;
}

void CKernel::add_to_normal(INT idx, DREAL weight)
{
	CIO::message(M_ERROR, "kernel does not support linadd optimization, add_to_normal not implemented\n") ;
}

void CKernel::clear_normal()
{
	CIO::message(M_ERROR, "kernel does not support linadd optimization, clear_normal not implemented\n") ;
}

INT CKernel::get_num_subkernels()
{
	return 1;
}

void CKernel::compute_by_subkernel(INT idx, DREAL * subkernel_contrib)
{
	CIO::message(M_ERROR, "kernel compute_by_subkernel not implemented\n") ;
}

const DREAL* CKernel::get_subkernel_weights(INT &num_weights)
{
	num_weights=1 ;
	return &combined_kernel_weight ;
}

void CKernel::set_subkernel_weights(DREAL* weights, INT num_weights)
{
	combined_kernel_weight = weights[0] ;
	if (num_weights!=1)
		CIO::message(M_ERROR, "number of subkernel weights should be one ...\n") ;
}

void CKernel::do_precompute_matrix()
{
	INT num_left=get_lhs()->get_num_vectors();
	INT num_right=get_rhs()->get_num_vectors();
	CIO::message(M_INFO, "precomputing kernel matrix (%ix%i)\n", num_left, num_right) ;

	ASSERT(num_left == num_right) ;
	ASSERT(get_lhs()==get_rhs()) ;
	INT num=num_left ;
	
	delete[] precomputed_matrix ;
	precomputed_matrix=new SHORTREAL[num*(num+1)/2] ;
	ASSERT(precomputed_matrix!=NULL) ;

	for (INT i=0; i<num; i++)
	{
		CIO::progress(i*i,0,num*num);
		for (INT j=0; j<=i; j++)
			precomputed_matrix[i*(i+1)/2+j] = compute(i,j) ;
	}

	CIO::progress(num*num,0,num*num);
	CIO::message(M_INFO, "\ndone.\n") ;
}
