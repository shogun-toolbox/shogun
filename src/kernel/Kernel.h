/************************************************************************/
/*                                                                      */
/*   kernel.h                                                           */
/*                                                                      */
/*   User defined kernel function. Feel free to plug in your own.       */
/*                                                                      */
/*                                                                      */
/************************************************************************/

/* KERNEL_PARM is defined in svm_common.h The field 'custom' is reserved for */
/* parameters of the user defined kernel. You can also access and use */
/* the parameters of the other kernels. */

#ifndef _KERNEL_H___
#define _KERNEL_H___
#include "lib/common.h"

class CKernel
{

	typedef struct kernel_cache {
		long   *index;  /* cache some kernel evalutations */
		CFLOAT *buffer; /* to improve speed */
		long   *invindex;
		long   *active2totdoc;
		long   *totdoc2active;
		long   *lru;
		long   *occu;
		long   elems;
		long   max_elems;
		long   time;
		long   activenum;
		long   buffsize;
	} KERNEL_CACHE;

	typedef struct kernel_parm {
		long    kernel_type;   /* 0=linear, 1=poly, 2=rbf, 3=sigmoid, 4=custom */
		long    poly_degree;
		double  rbf_gamma;
		double  coef_lin;
		double  coef_const;
		char    custom[50];    /* for user supplied kernel */
	} KERNEL_PARM;

	/* cache kernel evalutations to improve speed */
	void   get_kernel_row(KERNEL_CACHE *,DOC *, long, long, long *, CFLOAT *, 
			KERNEL_PARM *);
	void   cache_kernel_row(KERNEL_CACHE *,DOC *, long, KERNEL_PARM *);
	void   cache_multiple_kernel_rows(KERNEL_CACHE *,DOC *, long *, long, 
			KERNEL_PARM *);
	void   kernel_cache_shrink(KERNEL_CACHE *,long, long, long *);
	void   kernel_cache_init(KERNEL_CACHE *,long, long);
	void   kernel_cache_reset_lru(KERNEL_CACHE *);
	void   kernel_cache_cleanup(KERNEL_CACHE *);
	long   kernel_cache_malloc(KERNEL_CACHE *);
	void   kernel_cache_free(KERNEL_CACHE *,long);
	long   kernel_cache_free_lru(KERNEL_CACHE *);
	CFLOAT *kernel_cache_clean_and_malloc(KERNEL_CACHE *,long);
	long   kernel_cache_touch(KERNEL_CACHE *,long);
	long   kernel_cache_check(KERNEL_CACHE *,long);

	void tester(KERNEL_PARM *kernel_parm);
	double find_normalizer(KERNEL_PARM *kernel_parm, int num);
	double linear_top_kernel(KERNEL_PARM *kernel_parm, DOC* a, DOC* b);
	double top_kernel(KERNEL_PARM *kernel_parm, DOC* a, DOC* b);
	double cached_top_kernel(KERNEL_PARM *kernel_parm, DOC* a, DOC* b);
	double cached_fisher_kernel(KERNEL_PARM *kernel_parm, DOC* a, DOC* b);
	CFLOAT kernel(KERNEL_PARM *, DOC *, DOC *); 
	bool save_kernel(FILE* dest, CObservation* obs, int kernel_type);
};
#endif
