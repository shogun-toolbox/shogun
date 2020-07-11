/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn, Yuyu Zhang, Bjoern Esser,
 *          Viktor Gal
 */

#ifndef _COMMWORDSTRINGKERNEL_H___
#define _COMMWORDSTRINGKERNEL_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/mathematics/Math.h>
#include <shogun/kernel/string/StringKernel.h>

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
class CommWordStringKernel : public StringKernel<uint16_t>
{
	friend class VarianceKernelNormalizer;
	friend class SqrtDiagKernelNormalizer;
	friend class AvgDiagKernelNormalizer;
	friend class RidgeKernelNormalizer;
	friend class FirstElementKernelNormalizer;
	friend class TanimotoKernelNormalizer;
	friend class DiceKernelNormalizer;

	public:
		/** default constructor  */
		CommWordStringKernel();

		/** constructor
		 *
		 * @param size cache size
		 * @param use_sign if sign shall be used
		 */
		CommWordStringKernel(int32_t size, bool use_sign);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param use_sign if sign shall be used
		 * @param size cache size
		 */
		CommWordStringKernel(
			const std::shared_ptr<StringFeatures<uint16_t>>& l, const std::shared_ptr<StringFeatures<uint16_t>>& r,
			bool use_sign=false, int32_t size=10);

		~CommWordStringKernel() override;

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r) override;

		/** clean up kernel */
		void cleanup() override;

		/** return what type of kernel we are
		 *
		 * @return kernel type COMMWORDSTRING
		 */
		EKernelType get_kernel_type() override { return K_COMMWORDSTRING; }

		/** return the kernel's name
		 *
		 * @return name CommWordString
		 */
		const char* get_name() const override { return "CommWordStringKernel"; }

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
		bool init_optimization(
			int32_t count, int32_t *IDX, float64_t* weights) override;

		/** delete optimization
		 *
		 * @return if deleting was successful
		 */
		bool delete_optimization() override;

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

		/** clear normal */
		void clear_normal() override;

		/** return feature type the kernel can deal with
		 *
		 * @return feature type WORD
		 */
		EFeatureType get_feature_type() override { return F_WORD; }

		/** get dictionary
		 *
		 * @return dictionary weights
		 */
		SGVector<float64_t> get_dictionary() const
		{
			return dictionary_weights;
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
		float64_t compute(int32_t idx_a, int32_t idx_b) override
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
		/** dictionary weights - array to hold counters for all possible
		 * strings */
		SGVector<float64_t> dictionary_weights;

		/** if sign shall be used */
		bool use_sign;

		/** whether diagonal optimization shall be used */
		bool use_dict_diagonal_optimization;
		/** array to hold counters for all strings */
		int32_t* dict_diagonal_optimization;
};
}
#endif /* _COMMWORDSTRINGKERNEL_H__ */
