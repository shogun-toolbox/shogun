#ifndef _KERNEL_H___
#define _KERNEL_H___

#include "lib/common.h"
#include "features/Features.h"

#include <stdio.h>

class CKernel
{
	public:
		CKernel(LONG size);
		virtual ~CKernel();

		/** get kernel function for lhs feature vector x 
		  and rhs feature vector y
		 */
		REAL kernel(INT x, INT y);

		/** initialize kernel cache
		 *  make sure to check that your kernel can deal with the
		 *  supplied features (!)
		 *  set do_init to true if you want the kernel to call its setup function (like getting scaling parameters,...)
		 */
		virtual bool init(CFeatures* lhs, CFeatures* rhs, bool do_init);

		/// clean up your kernel
		virtual void cleanup()=0;

		/// load and save the kernel matrix
		bool load(CHAR* fname);
		bool save(CHAR* fname);

		/// load and save kernel init_data
		virtual bool load_init(FILE* src)=0;
		virtual bool save_init(FILE* dest)=0;
		
		/// get left/right hand side of features used in kernel
		inline CFeatures* get_lhs() { return lhs; }
		inline CFeatures* get_rhs() { return rhs; }

		// return what type of kernel we are Linear,Polynomial, Gaussian,...
		virtual EKernelType get_kernel_type()=0 ;

		/** return feature type the kernel can deal with
		  */
		virtual EFeatureType get_feature_type()=0;

		/** return feature class the kernel can deal with
		  */
		virtual EFeatureClass get_feature_class()=0;

		// return the name of a kernel
		virtual const CHAR* get_name()=0 ;

		// return the size of the kernel cache
		int get_cache_size() { return cache_size; }

		void get_kernel_row(LONG docnum, LONG *active2dnum, REAL *buffer) ;
		void cache_kernel_row(LONG x);
		void cache_multiple_kernel_rows(LONG* key, LONG varnum);
		void kernel_cache_reset_lru();

		/// set the time used for lru	
		inline void set_time(LONG t)
		{
			kernel_cache.time=t;
		}

		// Update lru time to avoid removal from cache.
		LONG kernel_cache_touch(LONG cacheidx)
		{
			if(kernel_cache.index[cacheidx] != -1)
			{
				kernel_cache.lru[kernel_cache.index[cacheidx]]=kernel_cache.time; 
				return(1);
			}
			return(0);
		}

	protected:
		/// compute kernel function for features a and b
		/// idx_{a,b} denote the index of the feature vectors
		/// in the corresponding feature object
		virtual REAL compute(INT x, INT y)=0;

		/**@ cache kernel evalutations to improve speed
		 */
		//@{
		void   kernel_cache_shrink(long, long, LONG *);

		/// init kernel cache of size megabytes
		void   kernel_cache_init(LONG size);
		void   kernel_cache_cleanup();
		LONG   kernel_cache_malloc();
		void   kernel_cache_free(LONG cacheidx);
		LONG   kernel_cache_free_lru();
		REAL *kernel_cache_clean_and_malloc(LONG);


		/// Is that row cached?
		inline LONG kernel_cache_check(LONG cacheidx)
		{
			return(kernel_cache.index[cacheidx] != -1);
		}
		//@}

		struct KERNEL_CACHE {
			LONG   *index;  
			REAL *buffer; 
			LONG   *invindex;
			LONG   *active2totdoc;
			LONG   *totdoc2active;
			LONG   *lru;
			LONG   *occu;
			LONG   elems;
			LONG   max_elems;
			LONG   time;
			LONG   activenum;
			LONG   buffsize;
			//			LONG   r_offs;
		};

	protected:
		/// kernel cache
		KERNEL_CACHE kernel_cache;

		/// cache_size in MB
		LONG cache_size;

		/// this *COULD* store the whole kernel matrix
		/// usually not applicable / faster
		REAL* kernel_matrix;

		/// feature vectors to occur on left hand side
		CFeatures* lhs;
		/// feature vectors to occur on right hand side
		CFeatures* rhs;
};
#endif
