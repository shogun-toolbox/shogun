#ifndef _KERNEL_H___
#define _KERNEL_H___
#include "lib/common.h"

class CKernel
{
public:
	CKernel();
	virtual ~CKernel();

	/// get kernel function for features a and b
	/// idx_{a,b} denote the index of the feature vectors
	/// in the corresponding feature object
	REAL kernel(CFeatures* a, int idx_a, CFeatures* b, int idx_b);

	virtual void init(CFeatures* f)=0;
	virtual void cleanup()=0;

	bool load(FILE* src);
	bool save(FILE* dest);

protected:
	/// compute kernel function for features a and b
	/// idx_{a,b} denote the index of the feature vectors
	/// in the corresponding feature object
	virtual REAL compute(CFeatures* a, int idx_a, CFeatures* b, int idx_b)=0;
	
	/**@ cache kernel evalutations to improve speed
	*/
	//@{
	void   get_kernel_row(DOC *, long, long, long *, REAL *);
	void   cache_kernel_row(DOC *, long);
	void   cache_multiple_kernel_rows(DOC *, long *, long);
	void   kernel_cache_shrink(long, long, long *);
	void   kernel_cache_init(long, long);
	void   kernel_cache_reset_lru();
	void   kernel_cache_cleanup();
	long   kernel_cache_malloc();
	void   kernel_cache_free(long);
	long   kernel_cache_free_lru();
	REAL *kernel_cache_clean_and_malloc(long);
	long   kernel_cache_touch(long);
	long   kernel_cache_check(long);
	//@}
	
	
	typedef struct kernel_cache {
		long   *index;  /* cache some kernel evalutations */
		REAL *buffer; /* to improve speed */
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

protected:
	long    kernel_type;   /* 0=linear, 1=poly, 2=rbf, 3=sigmoid, 4=custom */
	long    poly_degree;
	double  rbf_gamma;
	double  coef_lin;
	double  coef_const;
	char    custom[50];    /* for user supplied kernel */
};
#endif
