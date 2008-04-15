/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _COMMWORDSTRINGKERNEL_H___
#define _COMMWORDSTRINGKERNEL_H___

#include "lib/common.h"
#include "lib/Mathematics.h"
#include "kernel/StringKernel.h"

/** kernel CommWordString */
class CCommWordStringKernel: public CStringKernel<WORD>
{
	public:
		/** constructor
		 *
		 * @param size cache size
		 * @param use_sign if sign shall be used
		 * @param normalization_ type of normalization
		 */
		CCommWordStringKernel(INT size, bool use_sign,
			ENormalizationType normalization_=FULL_NORMALIZATION);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param use_sign if sign shall be used
		 * @param normalization_ type of normalization
		 * @param size cache size
		 */
		CCommWordStringKernel(
			CStringFeatures<WORD>* l, CStringFeatures<WORD>* r,
			bool use_sign=false,
			ENormalizationType normalization_=FULL_NORMALIZATION,
			INT size=10);

		virtual ~CCommWordStringKernel();

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool init(CFeatures* l, CFeatures* r);

		/** clean up kernel */
		virtual void cleanup();

		/** load kernel init_data
		 *
		 * @param src file to load from
		 * @return if loading was successful
		 */
		bool load_init(FILE* src);

		/** save kernel init_data
		 *
		 * @param dest file to save to
		 * @return if saving was successful
		 */
		bool save_init(FILE* dest);

		/** return what type of kernel we are
		 *
		 * @return kernel type COMMWORDSTRING
		 */
		virtual EKernelType get_kernel_type() { return K_COMMWORDSTRING; }

		/** return the kernel's name
		 *
		 * @return name CommWordString
		 */
		virtual const CHAR* get_name() { return "CommWordString"; }

		/** initialize dictionary
		 *
		 * @param size size
		 */
		virtual bool init_dictionary(INT size);

		/** initialize optimization
		 *
		 * @param count count
		 * @param IDX index
		 * @param weights weights
		 * @return if initializing was successful
		 */
		virtual bool init_optimization(INT count, INT *IDX,
			DREAL * weights);

		/** delete optimization
		 *
		 * @return if deleting was successful
		 */
		virtual bool delete_optimization();

		/** compute optimized
	 	*
	 	* @param idx index to compute
	 	* @return optimized value at given index
	 	*/
		virtual DREAL compute_optimized(INT idx);

		/** add to normal
		 *
		 * @param idx where to add
		 * @param weight what to add
		 */
		virtual void add_to_normal(INT idx, DREAL weight);

		/** clear normal */
		virtual void clear_normal();

		/** remove lhs from kernel */
		virtual void remove_lhs();

		/** remove rhs from kernel */
		virtual void remove_rhs();

		/** return feature type the kernel can deal with
		 *
		 * @return feature type WORD
		 */
		inline virtual EFeatureType get_feature_type() { return F_WORD; }

		/** get dictionary
		 *
		 * @param dsize dictionary size will be stored in here
		 * @param dweights dictionary weights will be stored in here
		 */
		void get_dictionary(INT& dsize, DREAL*& dweights)
		{
			dsize=dictionary_size;
			dweights = dictionary_weights;
		}

		/** compute scoring
		 *
		 * @param max_degree maximum degree
		 * @param num_feat number of features
		 * @param num_sym number of symbols
		 * @param target target
		 * @param num_suppvec number of support vectors
		 * @param IDX IDX
		 * @param alphas alphas
		 * @param do_init if initialization shall be performed
		 * @return computed scores
		 */
		virtual DREAL* compute_scoring(INT max_degree, INT& num_feat,
			INT& num_sym, DREAL* target, INT num_suppvec, INT* IDX,
			DREAL* alphas, bool do_init=true);

		/** compute consensus
		 *
		 * @param num_feat number of features
		 * @param num_suppvec number of support vectors
		 * @param IDX IDX
		 * @param alphas alphas
		 * @return computed consensus
		 */
		CHAR* compute_consensus(INT &num_feat, INT num_suppvec,
			INT* IDX, DREAL* alphas);


		/** set_use_dict_diagonal_optimization
		 *
		 * @param flag enable diagonal optimization
		 */
		void set_use_dict_diagonal_optimization(bool flag)
		{
			use_dict_diagonal_optimization=flag;
		}

		/** get.use.dict.diagonal.optimization
		 *
		 * @return true if diagonal optimization is on
		 */
		bool get_use_dict_diagonal_optimization()
		{
			return use_dict_diagonal_optimization;
		}
		
	protected:
		/** compute kernel function for features a and b
		 * idx_{a,b} denote the index of the feature vectors
		 * in the corresponding feature object
		 *
		 * @param idx_a index a
		 * @param idx_b index b
		 * @return computed kernel function at indices a,b
		 */
		inline virtual DREAL compute(INT idx_a, INT idx_b)
		{
			return compute_helper<false,false>(idx_a, idx_b);
		}

		/** helper for compute
		 * template arguments:
		 * do_sort if sorting shall be performed
		 * left_equal_right if lhs==rhs
		 *
		 * @param idx_a index a
		 * @param idx_b index b
		 * @return computed value
		 */
		template<bool do_sort, bool left_equal_right>
			DREAL compute_helper(INT idx_a, INT idx_b);

		/** normalize weight
		 *
		 * @param weights weights
		 * @param value value
		 * @param seq_num sequence number
		 * @param seq_len length of sequence
		 * @param p_normalization type of normalization
		 */
		inline DREAL normalize_weight(DREAL* weights, DREAL value,
			INT seq_num, INT seq_len,
			ENormalizationType p_normalization)
		{
			switch (p_normalization)
			{
				case NO_NORMALIZATION:
					return value;
					break;
				case SQRT_NORMALIZATION:
					return value/sqrt(weights[seq_num]);
					break;
				case FULL_NORMALIZATION:
					return value/weights[seq_num];
					break;
				case SQRTLEN_NORMALIZATION:
					return value/sqrt(sqrt((double) seq_len));
					break;
				case LEN_NORMALIZATION:
					return value/sqrt((double) seq_len);
					break;
				case SQLEN_NORMALIZATION:
					return value/seq_len;
					break;
				default:
					ASSERT(0);
			}

			return -CMath::INFTY;
		}

	protected:
		/** sqrt diagonal of left-hand side */
		DREAL *sqrtdiag_lhs;
		/** sqrt diagonal of right-hand side */
		DREAL *sqrtdiag_rhs;
		/** if kernel is initialized */
		bool initialized;

		/** size of dictionary (number of possible strings) */
		INT dictionary_size;
		/** dictionary weights - array to hold counters for all possible
		 * strings */
		DREAL* dictionary_weights;

		/** if sign shall be used */
		bool use_sign;
		/** type of normalization */
		ENormalizationType normalization;

		/** whether diagonal optimization shall be used */
		bool use_dict_diagonal_optimization;
		/** array to hold counters for all strings */
		INT* dict_diagonal_optimization;
};

#endif /* _COMMWORDSTRINGKERNEL_H__ */
