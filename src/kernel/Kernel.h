#ifndef _KERNEL_H___
#define _KERNEL_H___

#include "lib/common.h"
#include "features/Features.h"

#include <stdio.h>

class CKernel
{
public:
	CKernel();
	virtual ~CKernel();

	/// get kernel function for features a and b
	/// idx_{a,b} denote the index of the feature vectors
	/// in the corresponding feature object
	REAL kernel(CFeatures* a, long idx_a, CFeatures* b, long idx_b);

	/// initialize your kernel
	virtual void init(CFeatures* f)=0;
	/// clean up your kernel
	virtual void cleanup()=0;

	/// load and save the kernel
	bool load(FILE* src);
	bool save(FILE* dest);

	// check the feature object
	virtual bool check_features(CFeatures* f)=0 ;
	
	// return the name of a kernel
	virtual const char* get_name()=0 ;

	void get_kernel_row(CFeatures *train, long docnum,
			    long *active2dnum,  REAL *buffer) ;

protected:
	/// compute kernel function for features a and b
	/// idx_{a,b} denote the index of the feature vectors
	/// in the corresponding feature object
	virtual REAL compute(CFeatures* a, long idx_a, CFeatures* b, long idx_b)=0;
	
	/**@ cache kernel evalutations to improve speed
	*/
	//@{
	/*	void   get_kernel_row(DOC *, long, long, long *, REAL *);
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
		long   *index;  
		REAL *buffer; 
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
	*/
};
#endif
