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

#ifndef _WEIGHTEDCOMMWORDSTRINGKERNEL_H___
#define _WEIGHTEDCOMMWORDSTRINGKERNEL_H___

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/mathematics/Math.h>
#include <shogun/kernel/string/CommWordStringKernel.h>

namespace shogun
{
class CCommWordStringKernel;

/** @brief The WeightedCommWordString kernel may be used to compute the weighted
 * spectrum kernel (i.e. a spectrum kernel for 1 to K-mers, where each k-mer
 * length is weighted by some coefficient \f$\beta_k\f$) from strings that have
 * been mapped into unsigned 16bit integers.
 *
 * These 16bit integers correspond to k-mers. To applicable in this kernel they
 * need to be sorted (e.g. via the SortWordString pre-processor).
 *
 * It basically uses the algorithm in the unix "comm" command (hence the name)
 * to compute:
 *
 * \f[
 * k({\bf x},({\bf x'})= \sum_{k=1}^K\beta_k\Phi_k({\bf x})\cdot \Phi_k({\bf x'})
 * \f]
 *
 * where \f$\Phi_k\f$ maps a sequence \f${\bf x}\f$ that consists of letters in
 * \f$\Sigma\f$ to a feature vector of size \f$|\Sigma|^k\f$. In this feature
 * vector each entry denotes how often the k-mer appears in that \f${\bf x}\f$.
 *
 * Note that this representation is especially tuned to small alphabets
 * (like the 2-bit alphabet DNA), for which it enables spectrum kernels
 * of order 8.
 *
 * For this kernel the linadd speedups are quite efficiently implemented using
 * direct maps.
 *
 */
class CWeightedCommWordStringKernel: public CCommWordStringKernel
{
	public:
		/** default constructor  */
		CWeightedCommWordStringKernel();

		/** constructor
		 *
		 * @param size cache size
		 * @param use_sign if sign shall be used
		 */
		CWeightedCommWordStringKernel(int32_t size, bool use_sign);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param use_sign if sign shall be used
		 * @param size cache size
		 */
		CWeightedCommWordStringKernel(
			CStringFeatures<uint16_t>* l, CStringFeatures<uint16_t>* r,
			bool use_sign=false, int32_t size=10);

		virtual ~CWeightedCommWordStringKernel();

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool init(CFeatures* l, CFeatures* r);

		/** clean up kernel */
		virtual void cleanup();

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

		/** merge normal */
		void merge_normal();

		/** set weighted degree weights
		 *
		 * @return if setting was successful
		 */
		bool set_wd_weights();

		/** set custom weights (swig compatible)
		 *
		 * @param weights weights
		 * @return true if setting was successful
		 */
		bool set_weights(SGVector<float64_t> weights);

		/** return what type of kernel we are
		 *
		 * @return kernel type WEIGHTEDCOMMWORDSTRING
		 */
		virtual EKernelType get_kernel_type() { return K_WEIGHTEDCOMMWORDSTRING; }

		/** return the kernel's name
		 *
		 * @return name WeightedCommWordString
		 */
		virtual const char* get_name() const { return "WeightedCommWordStringKernel"; }

		/** return feature type the kernel can deal with
		 *
		 * @return feature type WORD
		 */
		virtual EFeatureType get_feature_type() { return F_WORD; }

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
		 * @return computed score
		 */
		virtual float64_t* compute_scoring(
			int32_t max_degree, int32_t& num_feat, int32_t& num_sym,
			float64_t* target, int32_t num_suppvec, int32_t* IDX,
			float64_t* alphas, bool do_init=true);

	protected:
		/** helper for compute
		 *
		 * @param idx_a index a
		 * @param idx_b index b
		 * @param do_sort if sorting shall be performed
		 */
		virtual float64_t compute_helper(
			int32_t idx_a, int32_t idx_b, bool do_sort);

	private:
		void init();

	protected:
		/** degree */
		int32_t degree;

		/** weights for each of the subkernels of degree 1...d */
		float64_t* weights;
};
}
#endif /* _WEIGHTEDCOMMWORDSTRINGKERNEL_H__ */
