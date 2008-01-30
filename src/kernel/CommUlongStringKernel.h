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

#ifndef _COMMULONGSTRINGKERNEL_H___
#define _COMMULONGSTRINGKERNEL_H___

#include "lib/common.h"
#include "lib/Mathematics.h"
#include "lib/DynamicArray.h"
#include "kernel/StringKernel.h"

/** kernel CommUlongString */
class CCommUlongStringKernel: public CStringKernel<ULONG>
{
	public:
		/** constructor
		 *
		 * @param size cache size
		 * @param use_sign if sign shall be used
		 * @param normalization_ type of normalization
		 */
		CCommUlongStringKernel(INT size=10, bool use_sign=false,
			ENormalizationType normalization_=FULL_NORMALIZATION);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param use_sign if sign shall be used
		 * @param normalization_ type of normalization
		 * @param size cache size
		 */
		CCommUlongStringKernel(
			CStringFeatures<ULONG>* l, CStringFeatures<ULONG>* r,
			bool use_sign=false,
			ENormalizationType normalization_=FULL_NORMALIZATION,
			INT size=10);

		virtual ~CCommUlongStringKernel();

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
		 * @return kernel type COMMULONGSTRING
		 */
		virtual EKernelType get_kernel_type() { return K_COMMULONGSTRING; }

		/** return the kernel's name
		 *
		 * @return name CommUlongString
		 */
		virtual const CHAR* get_name() { return "CommUlongString"; }

		/** initialize optimization
		 *
		 * @param count count
		 * @param IDX index
		 * @param weights weights
		 * @return if initializing was successful
		 */
		virtual bool init_optimization(INT count, INT* IDX, DREAL* weights);

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

		/** merge dictionaries
		 *
		 * @param t t
		 * @param j j
		 * @param k k
		 * @param vec vector
		 * @param dic dictionary
		 * @param dic_weights dictionary weights
		 * @param weight weight
		 * @param vec_idx vector index
		 * @param len length
		 * @param p_normalization normalization
		 */
		inline void merge_dictionaries(INT &t, INT j, INT &k,
			ULONG* vec, ULONG* dic, DREAL* dic_weights, DREAL weight,
			INT vec_idx, INT len, ENormalizationType p_normalization)
		{
			while (k<dictionary.get_num_elements() && dictionary[k] < vec[j-1])
			{
				dic[t]=dictionary[k];
				dic_weights[t]=dictionary_weights[k];
				t++;
				k++;
			}

			if (k<dictionary.get_num_elements() && dictionary[k]==vec[j-1])
			{
				dic[t]=vec[j-1];
				dic_weights[t]=dictionary_weights[k]+normalize_weight(weight, vec_idx, len, p_normalization);
				k++;
			}
			else
			{
				dic[t]=vec[j-1];
				dic_weights[t]=normalize_weight(weight, vec_idx, len, p_normalization);
			}
			t++;
		}

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
		 * @return feature type ULONG
		 */
		inline virtual EFeatureType get_feature_type() { return F_ULONG; }

		/** get dictionary
		 *
		 * @param dsize dictionary size will be stored in here
		 * @param dict dictionary will be stored in here
		 * @param dweights dictionary weights will be stored in here
		 */
		void get_dictionary(INT &dsize, ULONG*& dict, DREAL*& dweights) 
		{
			dsize=dictionary.get_num_elements();
			dict=dictionary.get_array();
			dweights = dictionary_weights.get_array();
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
		DREAL compute(INT idx_a, INT idx_b);

		/** normalize weight
		 *
		 * @param value value
		 * @param seq_num sequence number
		 * @param seq_len length of sequence
		 * @param p_normalization type of normalization
		 */
		inline DREAL normalize_weight(DREAL value, INT seq_num,
			INT seq_len, ENormalizationType p_normalization)
		{
			switch (p_normalization)
			{
				case NO_NORMALIZATION:
					return value;
					break;
				case SQRT_NORMALIZATION:
					return value/sqrt(sqrtdiag_lhs[seq_num]);
					break;
				case FULL_NORMALIZATION:
					return value/sqrtdiag_lhs[seq_num];
					break;
				case SQRTLEN_NORMALIZATION:
					return value/sqrt(sqrt(seq_len));
					break;
				case LEN_NORMALIZATION:
					return value/sqrt(seq_len);
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

		/** dictionary */
		CDynamicArray<ULONG> dictionary;
		/** dictionary weights */
		CDynamicArray<DREAL> dictionary_weights;

		/** if sign shall be used */
		bool use_sign;
		/** type of normalization */
		ENormalizationType normalization;
};

#endif /* _COMMULONGFSTRINGKERNEL_H__ */
