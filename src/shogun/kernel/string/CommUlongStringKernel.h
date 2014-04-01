/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _COMMULONGSTRINGKERNEL_H___
#define _COMMULONGSTRINGKERNEL_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/DynamicArray.h>
#include <shogun/kernel/string/StringKernel.h>

namespace shogun
{
template <class T> class CDynamicArray;
template <class ST> class CStringFeatures;

/** @brief The CommUlongString kernel may be used to compute the spectrum kernel
 * from strings that have been mapped into unsigned 64bit integers.
 *
 * These 64bit integers correspond to k-mers. To be applicable in this kernel
 * they need to be sorted (e.g. via the SortUlongString pre-processor).
 *
 * It basically uses the algorithm in the unix "comm" command (hence the name)
 * to compute:
 *
 * \f[
 * k({\bf x},({\bf x'})= \Phi_k({\bf x})\cdot \Phi_k({\bf x'})
 * \f]
 *
 * where \f$\Phi_k\f$ maps a sequence \f${\bf x}\f$ that consists of letters in
 * \f$\Sigma\f$ to a feature vector of size \f$|\Sigma|^k\f$. In this feature
 * vector each entry denotes how often the k-mer appears in that \f${\bf x}\f$.
 *
 * Note that this representation enables spectrum kernels of order 8 for 8bit
 * alphabets (like binaries) and order 32 for 2-bit alphabets like DNA.
 *
 * For this kernel the linadd speedups are implemented (though there is room for
 * improvement here when a whole set of sequences is ADDed) using sorted lists.
 *
 */
class CCommUlongStringKernel: public CStringKernel<uint64_t>
{
	public:
		/** constructor
		 *
		 * @param size cache size
		 * @param use_sign if sign shall be used
		 */
		CCommUlongStringKernel(int32_t size=10, bool use_sign=false);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param use_sign if sign shall be used
		 * @param size cache size
		 */
		CCommUlongStringKernel(
			CStringFeatures<uint64_t>* l, CStringFeatures<uint64_t>* r,
			bool use_sign=false,
			int32_t size=10);

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

		/** return what type of kernel we are
		 *
		 * @return kernel type COMMULONGSTRING
		 */
		virtual EKernelType get_kernel_type() { return K_COMMULONGSTRING; }

		/** return the kernel's name
		 *
		 * @return name CommUlongString
		 */
		virtual const char* get_name() const { return "CommUlongStringKernel"; }

		/** initialize optimization
		 *
		 * @param count count
		 * @param IDX index
		 * @param weights weights
		 * @return if initializing was successful
		 */
		virtual bool init_optimization(
			int32_t count, int32_t* IDX, float64_t* weights);

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
		virtual float64_t compute_optimized(int32_t idx);

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
		 */
		inline void merge_dictionaries(
			int32_t& t, int32_t j, int32_t& k, uint64_t* vec, SGVector<uint64_t> dic,
			SGVector<float64_t> dic_weights, float64_t weight, int32_t vec_idx)
		{
			while (k<dictionary.vlen && dictionary[k] < vec[j-1])
			{
				dic[t]=dictionary[k];
				dic_weights[t]=dictionary_weights[k];
				t++;
				k++;
			}

			if (k<dictionary.vlen && dictionary[k]==vec[j-1])
			{
				dic[t]=vec[j-1];
				dic_weights[t]=dictionary_weights[k]+normalizer->normalize_lhs(weight, vec_idx);
				k++;
			}
			else
			{
				dic[t]=vec[j-1];
				dic_weights[t]=normalizer->normalize_lhs(weight, vec_idx);
			}
			t++;
		}

		/** add to normal
		 *
		 * @param idx where to add
		 * @param weight what to add
		 */
		virtual void add_to_normal(int32_t idx, float64_t weight);

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
		virtual EFeatureType get_feature_type() { return F_ULONG; }

		/** get dictionary
		 *
		 * @param dsize dictionary size will be stored in here
		 * @param dict dictionary will be stored in here
		 * @param dweights dictionary weights will be stored in here
		 */
		void get_dictionary(
			int32_t &dsize, uint64_t*& dict, float64_t*& dweights)
		{
			dsize=dictionary.vlen;
			dict=dictionary.vector;
			dweights = dictionary_weights.vector;
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
		float64_t compute(int32_t idx_a, int32_t idx_b);

	protected:
		/** dictionary */
		SGVector<uint64_t> dictionary;
		/** dictionary weights */
		SGVector<float64_t> dictionary_weights;

		/** if sign shall be used */
		bool use_sign;
};
}
#endif /* _COMMULONGFSTRINGKERNEL_H__ */
