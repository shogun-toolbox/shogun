#include "lib/common.h"
#include "lib/io.h"
#include "lib/File.h"
#include "kernel/Kernel.h"
#include "features/Features.h"

#include <string.h>
#include <assert.h>

CKernel::CKernel(LONG size) : kernel_matrix(NULL), lhs(NULL), rhs(NULL), combined_kernel_weight(1)
{
	if (size<10)
		size=10;

	cache_size=size;
	CIO::message(M_INFO, "using a kernel cache of size %i MB\n", size) ;
	memset(&kernel_cache, 0x0, sizeof(KERNEL_CACHE));
}

CKernel::~CKernel()
{
	kernel_cache_cleanup();
}

/* calculate the kernel function */
REAL CKernel::kernel(INT idx_a, INT idx_b)
{
	if (idx_a < 0 || idx_b <0)
	{
		return 0;
	}

	return compute(idx_a, idx_b);
}

bool CKernel::init(CFeatures* l, CFeatures* r, bool do_init)
{
	//make sure features were indeed supplied
	assert(l);
	assert(r);

	//make sure features are compatible
	assert(l->get_feature_class() == r->get_feature_class());
	assert(l->get_feature_type() == r->get_feature_type());

	lhs=l;
	rhs=r;

	// allocate kernel cache but clean up beforehand
	kernel_cache_cleanup();
	kernel_cache_init(cache_size);

	return true;
}

/****************************** Cache handling *******************************/

void CKernel::kernel_cache_init(LONG buffsize)
{
	LONG i;
	LONG totdoc=get_lhs()->get_num_vectors();

	kernel_cache.index = new long[totdoc];
	kernel_cache.occu = new long[totdoc];
	kernel_cache.lru = new long[totdoc];
	kernel_cache.invindex = new long[totdoc];
	kernel_cache.active2totdoc = new long[totdoc];
	kernel_cache.totdoc2active = new long[totdoc];
	kernel_cache.buffer = new REAL[buffsize*1024*1024/sizeof(REAL)];

	kernel_cache.buffsize=(LONG)(buffsize*1024*1024/sizeof(REAL));

	kernel_cache.max_elems=(LONG)(kernel_cache.buffsize/totdoc);

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

void CKernel::get_kernel_row(LONG docnum, LONG *active2dnum, REAL *buffer)     
{
	LONG i,j,start;

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
void CKernel::cache_kernel_row(LONG m)
{
	register LONG j,k,l;
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
void CKernel::cache_multiple_kernel_rows(LONG* rows, LONG num_rows)
{
	// fill up kernel cache 
	for(LONG i=0;i<num_rows;i++) 
		cache_kernel_row(rows[i]);
}

// remove numshrink columns in the cache
// which correspond to examples marked
void CKernel::kernel_cache_shrink(LONG totdoc, LONG numshrink, LONG *after)
{                           
	register LONG i,j,jj,from=0,to=0,scount;     // 0 in after.
	LONG *keep;

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

	kernel_cache.max_elems=(LONG)(kernel_cache.buffsize/kernel_cache.activenum);
	if(kernel_cache.max_elems>totdoc) {
		kernel_cache.max_elems=totdoc;
	}

	delete[] keep;

}


void CKernel::kernel_cache_reset_lru()
{
	LONG maxlru=0,k;

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

LONG CKernel::kernel_cache_malloc()
{
	LONG i;

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

void CKernel::kernel_cache_free(LONG cacheidx)
{
	kernel_cache.occu[cacheidx]=0;
	kernel_cache.elems--;
}

// remove least recently used cache
// element
LONG CKernel::kernel_cache_free_lru()  
{                                     
	register LONG k,least_elem=-1,least_time;

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
REAL* CKernel::kernel_cache_clean_and_malloc(LONG cacheidx)
{             
	LONG result;
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
	return((REAL *)((LONG)kernel_cache.buffer
				+(kernel_cache.activenum*sizeof(REAL)*
					kernel_cache.index[cacheidx])));
}
bool CKernel::load(CHAR* fname)
{
	return false;
}

bool CKernel::save(CHAR* fname)
{
	INT i=0;
	INT num_left=get_lhs()->get_num_vectors();
	INT num_right=get_rhs()->get_num_vectors();
	LONG num_total=num_left*num_right;

	CFile f(fname, 'w', F_REAL);

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
		CIO::message(M_INFO, "kernel matrix of size %ld x %ld written (filesize: %ld)\n", num_left, num_right, num_total*sizeof(REAL));

    return (f.is_ok());
}


void CKernel::list_kernel()
{
	CIO::message(M_INFO, "0x%X - \"%s\" weight=%1.2f ", this, get_name(), get_combined_kernel_weight());
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
		case K_SPARSENORMSQUARED:
			CIO::message(M_INFO, "K_SPARSENORMSQUARED ");
			break;
		case K_COMBINED:
			CIO::message(M_INFO, "K_COMBINED ");
			break;
		default:
			CIO::message(M_ERROR, "ERROR ");
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
		default:
			CIO::message(M_ERROR, "ERROR ");
	}

	switch (get_feature_type())
	{
		case F_UNKNOWN:
			CIO::message(M_INFO, "F_UNKNOWN ");
			break;
		case F_REAL:
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
		default:
			CIO::message(M_ERROR, "ERROR ");
			break;
	}
	CIO::message(M_INFO, "\n");
}

