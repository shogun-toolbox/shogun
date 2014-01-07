/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _COMMWORDSTRINGKERNEL_H___
#define _COMMWORDSTRINGKERNEL_H___

#include <lib/common.h>
#include <mathematics/Math.h>
#include <kernel/string/StringKernel.h>

namespace shogun
{
/** @brief The CommWordString kernel may be used to compute the spectrum kernel
 * from strings that have been mapped into unsigned 16bit integers.
 *
 * These 16bit integers correspond to k-mers. To applicable in this kernel they
 * need to be sorted (e.g. via the SortWordString pre-processor).
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
 * Note that this representation is especially tuned to small alphabets
 * (like the 2-bit alphabet DNA), for which it enables spectrum kernels
 * of order up to 8.
 *
 * For this kernel the linadd speedups are quite efficiently implemented using
 * direct maps.
 *
 */
class CCommWordStringKernel : public CStringKernel<uint16_t>
{
	friend class CVarianceKernelNormalizer;
	friend class CSqrtDiagKernelNormalizer;
	friend class CAvgDiagKernelNormalizer;
	friend class CRidgeKernelNormalizer;
	friend class CFirstElementKernelNormalizer;
	friend class CTanimotoKernelNormalizer;
	friend class CDiceKernelNormalizer;

	public:
		/** default constructor  */
		CCommWordStringKernel();

		/** constructor
		 *
		 * @param size cache size
		 * @param use_sign if sign shall be used
		 */
		CCommWordStringKernel(int32_t size, bool use_sign);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param use_sign if sign shall be used
		 * @param size cache size
		 */
		CCommWordStringKernel(
			CStringFeatures<uint16_t>* l, CStringFeatures<uint16_t>* r,
			bool use_sign=false, int32_t size=10);

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

		/** return what type of kernel we are
		 *
		 * @return kernel type COMMWORDSTRING
		 */
		virtual EKernelType get_kernel_type() { return K_COMMWORDSTRING; }

		/** return the kernel's name
		 *
		 * @return name CommWordString
		 */
		virtual const char* get_name() const { return "CommWordStringKernel"; }

		/** initialize dictionary
		 *
		 * @param size size
		 */
		virtual bool init_dictionary(int32_t size);

		/** initialize optimization
		 *
		 * @param count count
		 * @param IDX index
		 * @param weights weights
		 * @return if initializing was successful
		 */
		virtual bool init_optimization(
			int32_t count, int32_t *IDX, float64_t* weights);

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

		/** add to normal
		 *
		 * @param idx where to add
		 * @param weight what to add
		 */
		virtual void add_to_normal(int32_t idx, float64_t weight);

		/** clear normal */
		virtual void clear_normal();

		/** return feature type the kernel can deal with
		 *
		 * @return feature type WORD
		 */
		virtual EFeatureType get_feature_type() { return F_WORD; }

		/** get dictionary
		 *
		 * @param dsize dictionary size will be stored in here
		 * @param dweights dictionary weights will be stored in here
		 */
		void get_dictionary(int32_t& dsize, float64_t*& dweights)
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
		virtual float64_t* compute_scoring(
			int32_t max_degree, int32_t& num_feat, int32_t& num_sym,
			float64_t* target, int32_t num_suppvec, int32_t* IDX,
			float64_t* alphas, bool do_init=true);

		/** compute consensus
		 *
		 * @param num_feat number of features
		 * @param num_suppvec number of support vectors
		 * @param IDX IDX
		 * @param alphas alphas
		 * @return computed consensus
		 */
		char* compute_consensus(
			int32_t &num_feat, int32_t num_suppvec, int32_t* IDX,
			float64_t* alphas);

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
		virtual float64_t compute(int32_t idx_a, int32_t idx_b)
		{
			return compute_helper(idx_a, idx_b, false);
		}

		/** helper for compute
		 *
		 * @param idx_a index a
		 * @param idx_b index b
		 * @param do_sort if sorting shall be performed
		 * @return computed value
		 */
		virtual float64_t compute_helper(
			int32_t idx_a, int32_t idx_b, bool do_sort);

		/** helper to compute only diagonal normalization for training
		 *
		 * @param idx_a index a
		 * @return unnormalized diagonal value
		 */
		virtual float64_t compute_diag(int32_t idx_a);

	private:
		void init();

	protected:
		/** size of dictionary (number of possible strings) */
		int32_t dictionary_size;
		/** dictionary weights - array to hold counters for all possible
		 * strings */
		float64_t* dictionary_weights;

		/** if sign shall be used */
		bool use_sign;

		/** whether diagonal optimization shall be used */
		bool use_dict_diagonal_optimization;
		/** array to hold counters for all strings */
		int32_t* dict_diagonal_optimization;
};
}
#endif /* _COMMWORDSTRINGKERNEL_H__ */
