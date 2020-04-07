/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Yuyu Zhang, Bjoern Esser, Sergey Lisitsyn
 */

#ifndef _WEIGHTEDCOMMWORDSTRINGKERNEL_H___
#define _WEIGHTEDCOMMWORDSTRINGKERNEL_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/mathematics/Math.h>
#include <shogun/kernel/string/CommWordStringKernel.h>

namespace shogun
{
class CommWordStringKernel;

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
class WeightedCommWordStringKernel: public CommWordStringKernel
{
	public:
		/** default constructor  */
		WeightedCommWordStringKernel();

		/** constructor
		 *
		 * @param size cache size
		 * @param use_sign if sign shall be used
		 */
		WeightedCommWordStringKernel(int32_t size, bool use_sign);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param use_sign if sign shall be used
		 * @param size cache size
		 */
		WeightedCommWordStringKernel(
			const std::shared_ptr<StringFeatures<uint16_t>>& l, const std::shared_ptr<StringFeatures<uint16_t>>& r,
			bool use_sign=false, int32_t size=10);

		~WeightedCommWordStringKernel() override;

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r) override;

		/** clean up kernel */
		void cleanup() override;

		/** compute optimized
		*
		* @param idx index to compute
		* @return optimized value at given index
		*/
		float64_t compute_optimized(int32_t idx) override;

		/** add to normal
		 *
		 * @param idx where to add
		 * @param weight what to add
		 */
		void add_to_normal(int32_t idx, float64_t weight) override;

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
		EKernelType get_kernel_type() override { return K_WEIGHTEDCOMMWORDSTRING; }

		/** return the kernel's name
		 *
		 * @return name WeightedCommWordString
		 */
		const char* get_name() const override { return "WeightedCommWordStringKernel"; }

		/** return feature type the kernel can deal with
		 *
		 * @return feature type WORD
		 */
		EFeatureType get_feature_type() override { return F_WORD; }

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
		float64_t* compute_scoring(
			int32_t max_degree, int32_t& num_feat, int32_t& num_sym,
			float64_t* target, int32_t num_suppvec, int32_t* IDX,
			float64_t* alphas, bool do_init=true) override;

	protected:
		/** helper for compute
		 *
		 * @param idx_a index a
		 * @param idx_b index b
		 * @param do_sort if sorting shall be performed
		 */
		float64_t compute_helper(
			int32_t idx_a, int32_t idx_b, bool do_sort) override;

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
