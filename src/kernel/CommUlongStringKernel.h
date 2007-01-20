/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Written (W) 1999-2007 Gunnar Raetsch
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _COMMULONGSTRINGKERNEL_H___
#define _COMMULONGSTRINGKERNEL_H___

#include "lib/common.h"
#include "lib/Mathematics.h"
#include "lib/DynamicArray.h"
#include "kernel/StringKernel.h"

class CCommUlongStringKernel: public CStringKernel<ULONG>
{
	public:
		CCommUlongStringKernel(INT size, bool use_sign, ENormalizationType normalization_=FULL_NORMALIZATION );
		CCommUlongStringKernel(CStringFeatures<ULONG>* l, CStringFeatures<ULONG>* r, bool use_sign=false, ENormalizationType normalization_=FULL_NORMALIZATION, INT size=10);
		~CCommUlongStringKernel();

		virtual bool init(CFeatures* l, CFeatures* r);
		virtual void cleanup();

		/// load and save kernel init_data
		bool load_init(FILE* src);
		bool save_init(FILE* dest);

		// return what type of kernel we are Linear,Polynomial, Gaussian,...
		virtual EKernelType get_kernel_type() { return K_COMMULONGSTRING; }

		// return the name of a kernel
		virtual const CHAR* get_name() { return "CommUlongString"; }

		virtual bool init_optimization(INT count, INT* IDX, DREAL* weights);
		virtual bool delete_optimization();
		virtual DREAL compute_optimized(INT idx);

		inline void merge_dictionaries(INT &t, INT j, INT &k, ULONG* vec, ULONG* dic, DREAL* dic_weights, DREAL weight, INT vec_idx, INT len, ENormalizationType p_normalization)
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

		virtual void add_to_normal(INT idx, DREAL weight);
		virtual void clear_normal();

		virtual void remove_lhs();
		virtual void remove_rhs();

		inline virtual EFeatureType get_feature_type() { return F_ULONG; }

		void get_dictionary(INT &dsize, ULONG*& dict, DREAL*& dweights) 
		{
			dsize=dictionary.get_num_elements();
			dict=dictionary.get_array();
			dweights = dictionary_weights.get_array();
		}

	protected:
		/// compute kernel function for features a and b
		/// idx_{a,b} denote the index of the feature vectors
		/// in the corresponding feature object
		DREAL compute(INT idx_a, INT idx_b);

		inline DREAL normalize_weight(DREAL value, INT seq_num, INT seq_len, ENormalizationType p_normalization)
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
		DREAL* sqrtdiag_lhs;
		DREAL* sqrtdiag_rhs;

		bool initialized;

		CDynamicArray<ULONG> dictionary;
		CDynamicArray<DREAL> dictionary_weights;

		bool use_sign;
		ENormalizationType normalization;
};
#endif
