#include "lib/common.h"
#include "lib/io.h"
#include "kernel/Kernel.h"
#include "features/Features.h"

#include <string.h>
#include <assert.h>

CKernel::CKernel(long size)
{
	if (size<100)
		size=100;

	cache_size=size;
	memset(&kernel_cache, 0x0, sizeof(KERNEL_CACHE));
}

CKernel::~CKernel()
{
}

/* calculate the kernel function */
REAL CKernel::kernel(long idx_a, long idx_b)
{
	if (idx_a < 0 || idx_b <0)
	{
#ifdef DEBUG
		printf("ERROR: (%d,%d)\n", idx_a, idx_b);
#endif
		return 0;
	}
//CIO::message("(%5d,%5d)\n", idx_a, idx_b);

	return compute(idx_a, idx_b);
}

void CKernel::init(CFeatures* l, CFeatures* r, bool do_init)
{
	assert(l!=0);
	assert(r!=0);
	lhs=l;
	rhs=r;

	// allocate kernel cache
	kernel_cache_init(cache_size);
}

/****************************** Cache handling *******************************/

void CKernel::kernel_cache_init(long buffsize)
{
	long i;
	long totdoc=lhs->get_num_vectors();

	kernel_cache.index = new long[totdoc];
	kernel_cache.occu = new long[totdoc];
	kernel_cache.lru = new long[totdoc];
	kernel_cache.invindex = new long[totdoc];
	kernel_cache.active2totdoc = new long[totdoc];
	kernel_cache.totdoc2active = new long[totdoc];
	kernel_cache.buffer = new REAL[buffsize*1024*1024/sizeof(REAL)];

	kernel_cache.buffsize=(long)(buffsize*1024*1024/sizeof(REAL));

	kernel_cache.max_elems=(long)(kernel_cache.buffsize/totdoc);
	//kernel_cache.r_offs=lhs->get_num_vectors();

	if(kernel_cache.max_elems>totdoc) {
		kernel_cache.max_elems=totdoc;
	}

	kernel_cache.elems=0;   // initialize cache 
	for(i=0;i<totdoc;i++) {
		kernel_cache.index[i]=-1;
		kernel_cache.lru[i]=0;
	}
	for(i=0;i<kernel_cache.max_elems;i++) {
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

void CKernel::get_kernel_row(long docnum, long *active2dnum, REAL *buffer)     
{
	long i,j,start;

#if 0
	for(i=0;(j=active2dnum[i])>=0;i++)
	{
		buffer[j]=(REAL)kernel(docnum, j);
	}
#else
	/* is cached? */
	if(kernel_cache.index[docnum] != -1) 
	{
		kernel_cache.lru[kernel_cache.index[docnum]]=kernel_cache.time; /* lru */
		start=kernel_cache.activenum*kernel_cache.index[docnum];

		for(i=0;(j=active2dnum[i])>=0;i++)
		{
			if(kernel_cache.totdoc2active[j] >= 0)
				buffer[j]=kernel_cache.buffer[start+kernel_cache.totdoc2active[j]];
			else
				buffer[j]=(REAL) kernel(docnum, j);
		}
	}
	else 
	{
		for(i=0;(j=active2dnum[i])>=0;i++)
			buffer[j]=(REAL) kernel(docnum, j);
	}
#endif
}


// Fills cache for the row m
void CKernel::cache_kernel_row(long m)
{
	register long j,k,l;
	register REAL *cache;

	if(!kernel_cache_check(m))   // not cached yet
	{
		cache = kernel_cache_clean_and_malloc(m);
		if(cache) {
			l=kernel_cache.totdoc2active[m];

			for(j=0;j<kernel_cache.activenum;j++)  // fill cache 
			{
				k=kernel_cache.active2totdoc[j];

				if((kernel_cache.index[k] != -1) && (l != -1) && (k != m)) {
					cache[j]=kernel_cache.buffer[kernel_cache.activenum
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

// Fills cache for the rows in key 
void CKernel::cache_multiple_kernel_rows(long* rows, long num_rows)
{
	// fill up kernel cache 
	for(int i=0;i<num_rows;i++) 
		cache_kernel_row(rows[i]);
}

// remove numshrink columns in the cache
// which correspond to examples marked
void CKernel::kernel_cache_shrink(long totdoc, long numshrink, long *after)
{                           
	register long i,j,jj,from=0,to=0,scount;     // 0 in after.
	long *keep;

	keep=new long[totdoc];
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

	kernel_cache.max_elems=(long)(kernel_cache.buffsize/kernel_cache.activenum);
	if(kernel_cache.max_elems>totdoc) {
		kernel_cache.max_elems=totdoc;
	}

	delete[] keep;

}


void CKernel::kernel_cache_reset_lru()
{
	long maxlru=0,k;

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

long CKernel::kernel_cache_malloc()
{
	long i;

	if(kernel_cache.elems < kernel_cache.max_elems) {
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

void CKernel::kernel_cache_free(long cacheidx)
{
	kernel_cache.occu[cacheidx]=0;
	kernel_cache.elems--;
}

// remove least recently used cache
// element
long CKernel::kernel_cache_free_lru()  
{                                     
	register long k,least_elem=-1,least_time;

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
REAL* CKernel::kernel_cache_clean_and_malloc(long cacheidx)
{             
	long result;
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
	return((REAL *)((long)kernel_cache.buffer
				+(kernel_cache.activenum*sizeof(REAL)*
					kernel_cache.index[cacheidx])));
}

bool CKernel::save(FILE* dest)
{
	//#ifdef USE_KERNEL_CACHE
	//	kernel_cache_init(&mykernel_cache,totdoc,100);
	//#else
	//	kernel_cache_init(&mykernel_cache,totdoc, 2);
	//#endif
	//
	//	DOC a;
	//	DOC b;
	//	for (int i=0; i<totdoc; i++)
	//	{
	//		a.docnum=i;
	//		for (int j=0; j<totdoc; j++)
	//		{
	//			b.docnum=j;
	//			double d=kernel(&mykernel_parm, &a, &b);
	//			fwrite(&d, sizeof(double),1, dest);
	//		}
	//	}
	//
	//	kernel_cache_cleanup(&mykernel_cache);
	//	return true;
	return false;
}
