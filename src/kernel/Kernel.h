/*
 * EXCEPT FOR THE KERNEL CACHING FUNCTIONS WHICH ARE (W) THORSTEN JOACHIMS
 * COPYRIGHT (C) 1999  UNIVERSITAET DORTMUND - ALL RIGHTS RESERVED
 *
 * this program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _KERNEL_H___
#define _KERNEL_H___

#include "lib/common.h"
#include "base/SGObject.h"
#include "features/Features.h"

class CSVM ;

/** class Kernel */
class CKernel : public CSGObject
{
	public:
		/** constructor
		 *
		 * @param size cache size
		 */
		CKernel(INT size);

		/** constructor
		 *
		 * @param l features for left-hand side
		 * @param r features for right-hand side
		 * @param size cache size
		 */
		CKernel(CFeatures* l, CFeatures* r, INT size);

		virtual ~CKernel();

		/** get kernel function for lhs feature vector a
		 * and rhs feature vector b
		 *
		 * @param idx_a index of feature vector a
		 * @param idx_b index of feature vector b
		 * @return computed kernel function
		 */
		inline DREAL kernel(INT idx_a, INT idx_b)
		{
			if (idx_a < 0 || idx_b <0)
				return 0;

			ASSERT(lhs!=NULL);
			ASSERT(rhs!=NULL);

			if (lhs==rhs)
			{
				INT num_vectors = lhs->get_num_vectors();

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

		/** get kernel matrix
		 *
		 * @param dst destination where matrix will be stored
		 * @param m dimension m of matrix
		 * @param n dimension n of matrix
		 */
		void get_kernel_matrix(DREAL** dst, INT* m, INT* n);

		/** get kernel matrix real
		 *
		 * @param m dimension m of matrix
		 * @param n dimension n of matrix
		 * @param target the kernel matrix
		 * @return the kernel matrix
		 */
		virtual DREAL* get_kernel_matrix_real(int &m, int &n, DREAL* target);

		/** get kernel matrix shortreal
		 *
		 * @param m dimension m of matrix
		 * @param n dimension n of matrix
		 * @param target target for kernel matrix
		 * @return the kernel matrix
		 */
		virtual SHORTREAL* get_kernel_matrix_shortreal(int &m, int &n,
			SHORTREAL* target);

		/** initialize kernel
		 *  e.g. setup lhs/rhs of kernel, precompute normalization
		 *  constants etc.
		 *  make sure to check that your kernel can deal with the
		 *  supplied features (!)
		 *
		 *  @param lhs features for left-hand side
		 *  @param rhs features for right-hand side
		 *  @return if init was successful
		 */
		virtual bool init(CFeatures* lhs, CFeatures* rhs);

		/** clean up your kernel
		 *
		 * abstract base method
		 */
		virtual void cleanup()=0;

		/** load the kernel matrix
		 *
		 * @param fname filename to load from
		 * @return if loading was succesful
		 */
		bool load(CHAR* fname);

		/** save kernel matrix
		 *
		 * @param fname filename to save to
		 * @return if saving was successful
		 */
		bool save(CHAR* fname);

		/** load kernel init_data
		 *
		 * abstract base method
		 *
		 * @param src file to load from
		 * @return if loading was successful
		 */
		virtual bool load_init(FILE* src)=0;

		/** save kernel init_data
		 *
		 * abstract base method
		 *
		 * @param dest file to save to
		 * @return if saving was successful
		 */
		virtual bool save_init(FILE* dest)=0;

		/** get left-hand side of features used in kernel
		 *
		 * @return features of left-hand side
		 */
		inline CFeatures* get_lhs() { SG_REF(lhs); return lhs; }

		/** get right-hand side of features used in kernel
		 *
		 * @return features of right-hand side
		 */
		inline CFeatures* get_rhs() { SG_REF(rhs); return rhs; }

		/** get number of vectors of lhs features
		 *
		 * @return number of vectors of left-hand side
		 */
		inline INT get_num_vec_lhs()
		{
			if (!lhs)
				return 0;
			else
				return lhs->get_num_vectors();
		}

		/** get number of vectors of rhs features
		 *
		 * @return number of vectors of right-hand side
		 */
		inline INT get_num_vec_rhs()
		{
			if (!rhs)
				return 0;
			else
				return rhs->get_num_vectors();
		}

		/** test whether features have been assigned to lhs and rhs
		 *
		 * @return true if features are assigned
		 */
		inline bool has_features()
		{
			return lhs && rhs;
		}

		/** remove lhs from kernel */
		virtual void remove_lhs();

		/** remove rhs from kernel */
		virtual void remove_rhs();

		/** return what type of kernel we are, e.g.
		 * Linear,Polynomial, Gaussian,...
		 *
		 * abstract base method
		 *
		 * @return kernel type
		 */
		virtual EKernelType get_kernel_type()=0 ;

		/** return feature type the kernel can deal with
		 *
		 * abstract base method
		 *
		 * @return feature type
		 */
		virtual EFeatureType get_feature_type()=0;

		/** return feature class the kernel can deal with
		 *
		 * abstract base method
		 *
		 * @return feature class
		 */
		virtual EFeatureClass get_feature_class()=0;

		/** get the name of a kernel
		 *
		 * @return name of kernel
		 */
		virtual const CHAR* get_name()=0 ;

		/** set the size of the kernel cache
		 *
		 * @param size of kernel cache
		 */
		inline void set_cache_size(INT size)
		{
			cache_size = size;
#ifdef USE_SVMLIGHT
			cache_reset();
#endif //USE_SVMLIGHT
		}

		/** return the size of the kernel cache
		 *
		 * @return size of kernel cache
		 */
		inline int get_cache_size() { return cache_size; }

#ifdef USE_SVMLIGHT
		/** cache reset */
		inline void cache_reset() { resize_kernel_cache(cache_size); }

		/** get maximum elements in cache
		 *
		 * @return maximum elements in cache
		 */
		inline int get_max_elems_cache() { return kernel_cache.max_elems; }

		/** get activenum cache
		 *
		 * @return activecnum cache
		 */
		inline int get_activenum_cache() { return kernel_cache.activenum; }

		/** get kernel row
		 *
		 * @param docnum docnum
		 * @param active2dnum active2dnum
		 * @param buffer buffer
		 * @param full_line full line
		 */
		void get_kernel_row(INT docnum, INT *active2dnum, DREAL *buffer,
			bool full_line=false);

		/** cache kernel row
		 *
		 * @param x x
		 */
		void cache_kernel_row(INT x);

		/** cache multiple kernel rows
		 *
		 * @param key key
		 * @param varnum
		 */
		void cache_multiple_kernel_rows(INT* key, INT varnum);

		/** kernel cache reset lru */
		void kernel_cache_reset_lru();

		/** kernel cache shrink
		 *
		 * @param totdoc totdoc
		 * @param num_shrink number of shrink
		 * @param after after
		 */
		void kernel_cache_shrink(INT totdoc, INT num_shrink, INT *after);

		/** resize kernel cache
		 *
		 * @param size new size
		 * @param regression_hack hack for regression
		 */
		void resize_kernel_cache(KERNELCACHE_IDX size,
			bool regression_hack=false);

		/** set the lru time
		 *
		 * @param t the time to use
		 */
		inline void set_time(INT t)
		{
			kernel_cache.time=t;
		}

		/** update lru time of item at given index to avoid removal from cache
		 *
		 * @param cacheidx index in cache
		 * @return if updating was successful
		 */
		inline INT kernel_cache_touch(INT cacheidx)
		{
			if(kernel_cache.index[cacheidx] != -1)
			{
				kernel_cache.lru[kernel_cache.index[cacheidx]]=kernel_cache.time;
				return(1);
			}
			return(0);
		}

		/** check if row at given index is cached
		 *
		 * @param cacheidx index in cache
		 * @return if row at given index is cached
		 */
		inline INT kernel_cache_check(INT cacheidx)
		{
			return(kernel_cache.index[cacheidx] >= 0);
		}

		/** check if there is room for one more row in kernel cache
		 *
		 * @return if there is room for one more row in kernel cache
		 */
		inline INT kernel_cache_space_available()
		{
			return(kernel_cache.elems < kernel_cache.max_elems);
		}

		void   kernel_cache_init(INT size, bool regression_hack=false);
		void   kernel_cache_cleanup();
#endif //USE_SVMLIGHT

		/** list kernel */
		void list_kernel();

		/** check if kernel has given property
		 *
		 * @param p kernel property
		 * @return if kernel has given property
		 */
		inline bool has_property(EKernelProperty p) { return (properties & p) != 0; }

		/** for optimizable kernels, i.e. kernels where the weight
		 * vector can be computed explicitely (if it fits into memory)
		 */
		virtual void clear_normal();

		/** add vector*factor to 'virtual' normal vector
		 *
		 * @param vector_idx index
		 * @param weight weight
		 */
		virtual void add_to_normal(INT vector_idx, DREAL weight);

		/** get optimization type
		 *
		 * @return optimization type
		 */
		inline EOptimizationType get_optimization_type() { return opt_type; }

		/** set optimization type
		 *
		 * @param t optimization type to set
		 */
		virtual inline void set_optimization_type(EOptimizationType t) { opt_type=t;}

		/** check if optimization is initialized
		 *
		 * @return if optimization is initialized
		 */
		inline bool get_is_initialized() { return optimization_initialized; }

		/** initialize optimization
		 *
		 * @param count count
		 * @param IDX index
		 * @param weights weights
		 * @return if initializing was successful
		 */
		virtual bool init_optimization(INT count, INT *IDX, DREAL *weights);

		/** delete optimization
		 *
		 * @return if deleting was successful
		 */
		virtual bool delete_optimization();

		/** initialize optimization
		 *
		 * @param svm svm model
		 * @return if initializing was successful
		 */
		bool init_optimization_svm(CSVM * svm) ;

		/** compute optimized
		 *
		 * @param vector_idx index to compute
		 * @return optimized value at given index
		 */
		virtual DREAL compute_optimized(INT vector_idx);

		/** computes output for a batch of examples in an optimized fashion
		 * (favorable if kernel supports it, i.e. has KP_BATCHEVALUATION.  to
		 * the outputvector target (of length num_vec elements) the output for
		 * the examples enumerated in vec_idx are added. therefore make sure
		 * that it is initialized with ZERO. the following num_suppvec, IDX,
		 * alphas arguments are the number of support vectors, their indices
		 * and weights
		 */
		virtual void compute_batch(INT num_vec, INT* vec_idx, DREAL* target, INT num_suppvec, INT* IDX, DREAL* alphas, DREAL factor=1.0);

		/** get combined kernel weight
		 *
		 * @return combined kernel weight
		 */
		inline DREAL get_combined_kernel_weight() { return combined_kernel_weight; }

		/** set combined kernel weight
		 *
		 * @param nw nw
		 */
		inline void set_combined_kernel_weight(double nw) { combined_kernel_weight=nw; }

		/** get number of subkernels
		 *
		 * @return number of subkernels
		 */
		virtual INT get_num_subkernels();

		/** compute by subkernel
		 *
		 * @param vector_idx index
		 * @param subkernel_contrib subkernel contribution
		 */
		virtual void compute_by_subkernel(INT vector_idx, DREAL * subkernel_contrib);

		/** get subkernel weights
		 *
		 * @param num_weights number of weights will be stored here
		 * @return subkernel weights
		 */
		virtual const DREAL* get_subkernel_weights(INT& num_weights);

		/** set subkernel weights
		 *
		 * @param weights subkernel weights
		 * @param num_weights number of weights
		 */
		virtual void set_subkernel_weights(DREAL* weights, INT num_weights);

		//FIXME: precompute matrix should be dropped, handling should be via customkernel
		/** get precompute matrix
		 *
		 * @return if matrix shall be precomputed
		 */
		inline bool get_precompute_matrix() { return precompute_matrix ;  }

		/** get precompute subkernel matrix
		 *
		 * @return if subkernel matrix shall be precomputed
		 */
		inline bool get_precompute_subkernel_matrix() { return precompute_subkernel_matrix ;  }

		/** set precompute matrix
		 *
		 * @param flag flag
		 * @param subkernel_flag subkernel flag
		 */
		inline virtual void set_precompute_matrix(bool flag, bool subkernel_flag)
		{
			precompute_matrix = flag;
			precompute_subkernel_matrix = subkernel_flag;

			if (!precompute_matrix)
			{
				delete[] precomputed_matrix;
				precomputed_matrix = NULL;
			}
		}

	protected:
		/** set property
		 *
		 * @param p kernel property to set
		 */
		inline void set_property(EKernelProperty p)
		{
			properties |= p;
		}

		/** unset property
		 *
		 * @param p kernel property to unset
		 */
		inline void unset_property(EKernelProperty p)
		{
			properties &= (properties | p) ^ p;
		}

		/** set is initialized
		 *
		 * @param p_init if optimization shall be set to initialized
		 */
		inline void set_is_initialized(bool p_init) { optimization_initialized=p_init; }

		/** compute kernel function for features a and b
		 * idx_{a,b} denote the index of the feature vectors
		 * in the corresponding feature object
		 *
		 * abstract base method
		 *
		 * @param x index a
		 * @param y index b
		 * @return computed kernel function at indices a,b
		 */
		virtual DREAL compute(INT x, INT y)=0;

		/// matrix precomputation
		void do_precompute_matrix();

		/** initialize sqrt diagonal
		 *
		 * @param v v
		 * @param num num
		 */
		void init_sqrt_diag(DREAL *v, INT num);

#ifdef USE_SVMLIGHT
		/**@ cache kernel evalutations to improve speed
		*/
		//@{
		struct KERNEL_CACHE {
			/** index */
			INT   *index;
			/** inverse index */
			INT   *invindex;
			/** active2totdoc */
			INT   *active2totdoc;
			/** totdoc2active */
			INT   *totdoc2active;
			/** least recently used */
			INT   *lru;
			/** occu */
			INT   *occu;
			/** elements */
			INT   elems;
			/** max elements */
			INT   max_elems;
			/** time */
			INT   time;
			/** active num */
			INT   activenum;

			/** buffer */
			KERNELCACHE_ELEM  *buffer;
			/** buffer size */
			KERNELCACHE_IDX   buffsize;
		};

		/** kernel thread parameters */
		struct S_KTHREAD_PARAM
		{
			/** kernel */
			CKernel* kernel;
			/** kernel cache */
			KERNEL_CACHE* kernel_cache;
			/** cache */
			KERNELCACHE_ELEM** cache;
			/** uncached rows */
			INT* uncached_rows;
			/** number of uncached rows */
			INT num_uncached;
			/** needs computation */
			BYTE* needs_computation;
			/** start */
			INT start;
			/** end */
			INT end;
		};
		static void* cache_multiple_kernel_row_helper(void* p);

		/// init kernel cache of size megabytes
		void   kernel_cache_free(INT cacheidx);
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

		/** precomputed matrix */
		SHORTREAL * precomputed_matrix;
		/** if subkernel matrix shall be precomputed */
		bool precompute_subkernel_matrix;
		/** if matrix shall be precomputed */
		bool precompute_matrix ;

		/// feature vectors to occur on left hand side
		CFeatures* lhs;
		/// feature vectors to occur on right hand side
		CFeatures* rhs;

		/** combined kernel weight */
		DREAL combined_kernel_weight;

		/** if optimization is initialized */
		bool optimization_initialized;
		/** optimization type (currently FASTBUTMEMHUNGRY and
		 * SLOWBUTMEMEFFICIENT)
		 */
		EOptimizationType opt_type;

		/** properties */
		ULONG  properties;
};

#endif /* _KERNEL_H__ */
