#ifndef _KERNEL_H___
#define _KERNEL_H___

#include "lib/common.h"
#include "features/Features.h"

typedef REAL CACHE_ELEM ;
typedef LONG CACHE_IDX ;

#include <stdio.h>

class CKernel
{
	public:
		CKernel(CACHE_IDX size);
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
		inline CFeatures* get_lhs() { return lhs; } ;
		inline CFeatures* get_rhs() { return rhs;  } ;

		inline void cache_reset() {	resize_kernel_cache(cache_size) ; } ;
		
		/// takes all necessary steps if the lhs is removed from kernel
		virtual void remove_lhs() { if (lhs) cache_reset() ; lhs = NULL ;  } ;
		/// takes all necessary steps if the rhs is removed from kernel
		virtual void remove_rhs() { if (rhs) cache_reset() ; rhs = NULL ;  } ;
		
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

		void get_kernel_row(CACHE_IDX docnum, LONG *active2dnum, REAL *buffer) ;
		void cache_kernel_row(CACHE_IDX x);
		void cache_multiple_kernel_rows(LONG* key, INT varnum);
		void kernel_cache_reset_lru();

		void resize_kernel_cache(CACHE_IDX size) ;
		
		/// set the time used for lru	
		inline void set_time(LONG t)
		{
			kernel_cache.time=t;
		}

		// Update lru time to avoid removal from cache.
		CACHE_IDX kernel_cache_touch(CACHE_IDX cacheidx)
		{
			if(kernel_cache.index[cacheidx] != -1)
			{
				kernel_cache.lru[kernel_cache.index[cacheidx]]=kernel_cache.time; 
				return(1);
			}
			return(0);
		}

		void list_kernel();

		/** for optimizable kernels, i.e. kernels where the weight 
		 * vector can be computed explicitely (if it fits into memory) 
		 */

		virtual bool init_optimization(INT count, INT *IDX, REAL * weights); 
		virtual void delete_optimization();
		virtual REAL compute_optimized(INT idx);
	
		inline bool get_is_initialized() { return optimization_initialized; }
		inline void set_is_initialized(bool init) { optimization_initialized=init; }

		bool is_optimizable() ;

		inline double get_combined_kernel_weight() { return combined_kernel_weight; }
		inline void set_combined_kernel_weight(double nw) { combined_kernel_weight=nw; }

	protected:
		/// compute kernel function for features a and b
		/// idx_{a,b} denote the index of the feature vectors
		/// in the corresponding feature object
		virtual REAL compute(INT x, INT y)=0;

		/**@ cache kernel evalutations to improve speed
		 */
		//@{
		void   kernel_cache_shrink(CACHE_IDX, CACHE_IDX, CACHE_IDX *);

		/// init kernel cache of size megabytes
		void   kernel_cache_init(CACHE_IDX size);
		void   kernel_cache_cleanup();
		CACHE_IDX   kernel_cache_malloc();
		void   kernel_cache_free(CACHE_IDX cacheidx);
		CACHE_IDX   kernel_cache_free_lru();
		CACHE_ELEM *kernel_cache_clean_and_malloc(CACHE_IDX);


		/// Is that row cached?
		inline CACHE_IDX kernel_cache_check(CACHE_IDX cacheidx)
		{
			return(kernel_cache.index[cacheidx] != -1);
		}
		//@}

		struct KERNEL_CACHE {
			CACHE_IDX   *index;  
			CACHE_ELEM  *buffer; 
			CACHE_IDX   *invindex;
			CACHE_IDX   *active2totdoc;
			CACHE_IDX   *totdoc2active;
			CACHE_IDX   *lru;
			CACHE_IDX   *occu;
			CACHE_IDX   elems;
			CACHE_IDX   max_elems;
			CACHE_IDX   time;
			CACHE_IDX   activenum;
			CACHE_IDX   buffsize;
			//			LONG   r_offs;
		};

	protected:
		/// kernel cache
		KERNEL_CACHE kernel_cache;

		/// cache_size in MB
		CACHE_IDX cache_size;

		/// this *COULD* store the whole kernel matrix
		/// usually not applicable / faster
		CACHE_ELEM* kernel_matrix;

		/// feature vectors to occur on left hand side
		CFeatures* lhs;
		/// feature vectors to occur on right hand side
		CFeatures* rhs;

		REAL combined_kernel_weight ;
	
		bool optimization_initialized ;
		
};
#endif
