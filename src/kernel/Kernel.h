/*
 * EXCEPT FOR THE KERNEL CACHING FUNCTIONS WHICH ARE (W) THORSTEN JOACHIMS
 * COPYRIGHT (C) 1999  UNIVERSITAET DORTMUND - ALL RIGHTS RESERVED
 *
 * this program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Written (W) 1999-2007 Gunnar Raetsch
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _KERNEL_H___
#define _KERNEL_H___

#include "lib/common.h"
#include "base/SGObject.h"
#include "features/Features.h"

class CKernel : public CSGObject
{
	public:
		CKernel(INT size);
		CKernel(CFeatures* lhs, CFeatures* rhs, INT size);
		virtual ~CKernel();

		/** get kernel function for lhs feature vector a 
		  and rhs feature vector b
		 */
		inline DREAL kernel(INT idx_a, INT idx_b)
		{
			if (idx_a < 0 || idx_b <0)
				return 0;

			if (lhs==rhs)
			{
				int num_vectors = lhs->get_num_vectors();

				if (idx_a>=num_vectors)
					idx_a=2*num_vectors-1-idx_a;

				if (idx_b>=num_vectors)
					idx_b=2*num_vectors-1-idx_b;
			}

			if (precompute_matrix && (precomputed_matrix==NULL) && (lhs==rhs))
				do_precompute_matrix() ;

			if (precompute_matrix && (precomputed_matrix!=NULL))
			{
				if (idx_a>=idx_b)
					return precomputed_matrix[idx_a*(idx_a+1)/2+idx_b] ;
				else
					return precomputed_matrix[idx_b*(idx_b+1)/2+idx_a] ;
			}

			return compute(idx_a, idx_b);
		}

		void get_kernel_matrix(DREAL** dst, INT* m, INT* n);
		virtual DREAL* get_kernel_matrix_real(int &m, int &n, DREAL* target);
		virtual SHORTREAL* get_kernel_matrix_shortreal(int &m, int &n, SHORTREAL* target);

		/** initialize kernel
		 *  e.g. setup lhs/rhs of kernel, precompute normalization constants etc.
		 *  make sure to check that your kernel can deal with the
		 *  supplied features (!)
		 */
		virtual bool init(CFeatures* lhs, CFeatures* rhs);

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

		/// takes all necessary steps if the lhs is removed from kernel
		virtual void remove_lhs();

		/// takes all necessary steps if the rhs is removed from kernel
		virtual void remove_rhs();
		
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
		inline int get_cache_size() { return cache_size; }
#ifdef USE_SVMLIGHT
		inline void cache_reset() {	resize_kernel_cache(cache_size) ; } ;
		inline int get_max_elems_cache() { return kernel_cache.max_elems; }
		inline int get_activenum_cache() { return kernel_cache.activenum; }
		void get_kernel_row(INT docnum, INT *active2dnum, DREAL *buffer, bool full_line=false) ;
		void cache_kernel_row(INT x);
		void cache_multiple_kernel_rows(INT* key, INT varnum);
		void kernel_cache_reset_lru();
		void kernel_cache_shrink(INT totdoc, INT num_shrink, INT *after);

		void resize_kernel_cache(KERNELCACHE_IDX size, bool regression_hack=false);
		
		/// set the time used for lru	
		inline void set_time(INT t)
		{
			kernel_cache.time=t;
		}

		// Update lru time to avoid removal from cache.
		inline INT kernel_cache_touch(INT cacheidx)
		{
			if(kernel_cache.index[cacheidx] != -1)
			{
				kernel_cache.lru[kernel_cache.index[cacheidx]]=kernel_cache.time; 
				return(1);
			}
			return(0);
		}

		/// Is that row cached?
		inline INT kernel_cache_check(INT cacheidx)
		{
			return(kernel_cache.index[cacheidx] >= 0);
		}

		inline INT kernel_cache_space_available()
			/* Is there room for one more row? */
		{
			return(kernel_cache.elems < kernel_cache.max_elems);
		}
#endif //USE_SVMLIGHT

		void list_kernel();

		inline bool has_property(EKernelProperty p) { return (properties & p) != 0; }

		/** for optimizable kernels, i.e. kernels where the weight 
		 * vector can be computed explicitely (if it fits into memory) 
		 */
		virtual void clear_normal();
		///add vector*factor to 'virtual' normal vector
		virtual void add_to_normal(INT idx, DREAL weight) ;

		inline EOptimizationType get_optimization_type() { return opt_type; }
		virtual inline void set_optimization_type(EOptimizationType t) { opt_type=t;}

		inline bool get_is_initialized() { return optimization_initialized; }
		virtual bool init_optimization(INT count, INT *IDX, DREAL * weights); 
		virtual bool delete_optimization();
		virtual DREAL compute_optimized(INT idx);

		/** computes output for a batch of examples in an optimized fashion (favorable if kernel supports it,
		 * i.e. has KP_BATCHEVALUATION.
		 * to the outputvector target (of length num_vec elements) the output for the examples enumerated
		 * in vec_idx are added. therefore make sure that it is initialized with ZERO. the following num_suppvec,
		 * IDX, alphas arguments are the number of support vectors, their indices and weights
		 */
		virtual void compute_batch(INT num_vec, INT* vec_idx, DREAL* target, INT num_suppvec, INT* IDX, DREAL* alphas, DREAL factor=1.0);
		
		inline double get_combined_kernel_weight() { return combined_kernel_weight; }
		inline void set_combined_kernel_weight(double nw) { combined_kernel_weight=nw; }
		virtual INT get_num_subkernels();
		virtual void compute_by_subkernel(INT idx, DREAL * subkernel_contrib);
		virtual const DREAL* get_subkernel_weights(INT& num_weights);
		virtual void set_subkernel_weights(DREAL* weights, INT num_weights);

		//fixme: precompute matrix should be dropped, handling should be via customkernel
		inline bool get_precompute_matrix() { return precompute_matrix ;  }
		inline bool get_precompute_subkernel_matrix() { return precompute_subkernel_matrix ;  }
		inline virtual void set_precompute_matrix(bool flag, bool subkernel_flag)
		{ 
			precompute_matrix = flag ; 
			precompute_subkernel_matrix = subkernel_flag ; 

			if (!precompute_matrix)
			{
				delete[] precomputed_matrix ;
				precomputed_matrix = NULL ;
			}
		}
		
	protected:

		inline void set_property(EKernelProperty p)
		{
			properties |= p;
		}

		inline void unset_property(EKernelProperty p)
		{
			properties &= (properties | p) ^ p;
		}

		inline void set_is_initialized(bool p_init) { optimization_initialized=p_init; }

		/// compute kernel function for features a and b
		/// idx_{a,b} denote the index of the feature vectors
		/// in the corresponding feature object
		virtual DREAL compute(INT x, INT y)=0;

		/// matrix precomputation
		void do_precompute_matrix() ;

		void init_sqrt_diag(DREAL *v, INT num);

#ifdef USE_SVMLIGHT
		/**@ cache kernel evalutations to improve speed
		 */
		//@{
		struct KERNEL_CACHE {
			INT   *index;  
			INT   *invindex;
			INT   *active2totdoc;
			INT   *totdoc2active;
			INT   *lru;
			INT   *occu;
			INT   elems;
			INT   max_elems;
			INT   time;
			INT   activenum;

			KERNELCACHE_ELEM  *buffer; 
			KERNELCACHE_IDX   buffsize;
		};

		struct S_KTHREAD_PARAM 
		{
			CKernel* kernel;
			KERNEL_CACHE* kernel_cache;
			KERNELCACHE_ELEM** cache;
			INT* uncached_rows;
			INT num_uncached;
			BYTE* needs_computation;
			INT start;
			INT end;
		};
		static void* cache_multiple_kernel_row_helper(void* p);

		/// init kernel cache of size megabytes
		void   kernel_cache_init(INT size, bool regression_hack=false);
		void   kernel_cache_free(INT cacheidx);
		void   kernel_cache_cleanup();
		INT   kernel_cache_malloc();
		INT   kernel_cache_free_lru();
		KERNELCACHE_ELEM *kernel_cache_clean_and_malloc(INT cacheidx);
#endif //USE_SVMLIGHT
		//@}


	protected:
		/// cache_size in MB
		INT cache_size;

#ifdef USE_SVMLIGHT
		/// kernel cache
		KERNEL_CACHE kernel_cache;
#endif //USE_SVMLIGHT

		/// this *COULD* store the whole kernel matrix
		/// usually not applicable / faster
		KERNELCACHE_ELEM* kernel_matrix;

		SHORTREAL * precomputed_matrix ;
		bool precompute_subkernel_matrix ;
		bool precompute_matrix ;

		/// feature vectors to occur on left hand side
		CFeatures* lhs;
		/// feature vectors to occur on right hand side
		CFeatures* rhs;

		DREAL combined_kernel_weight ;
	
		bool optimization_initialized ;
		/// optimization type (currently FASTBUTMEMHUNGRY and SLOWBUTMEMEFFICIENT)
		EOptimizationType opt_type;

		ULONG  properties;
};
#endif
