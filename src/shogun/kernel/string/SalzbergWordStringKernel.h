/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Yuyu Zhang, Bjoern Esser, Sergey Lisitsyn
 */

#ifndef _SALZBERGWORDSTRINGKERNEL_H___
#define _SALZBERGWORDSTRINGKERNEL_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/kernel/string/StringKernel.h>
#include <shogun/classifier/PluginEstimate.h>
#include <shogun/features/StringFeatures.h>

namespace shogun
{
/** @brief The SalzbergWordString kernel implements the Salzberg kernel.
 *
 * It is described in
 *
 * Engineering Support Vector Machine Kernels That Recognize Translation Initiation Sites
 * A. Zien, G.Raetsch, S. Mika, B. Schoelkopf, T. Lengauer, K.-R. Mueller
 *
 */
class SalzbergWordStringKernel: public StringKernel<uint16_t>
{
	public:
		/** default constructor  */
		SalzbergWordStringKernel();

		/** constructor
		 *
		 * @param size cache size
		 * @param pie the plugin estimate
		 * @param labels optional labels to set prior from
		 */
		SalzbergWordStringKernel(int32_t size, std::shared_ptr<PluginEstimate> pie, std::shared_ptr<Labels> labels=NULL);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param pie the plugin estimate
		 * @param labels optional labels to set prior from
		 */
		SalzbergWordStringKernel(
			std::shared_ptr<StringFeatures<uint16_t>> l, std::shared_ptr<StringFeatures<uint16_t>> r,
			std::shared_ptr<PluginEstimate >pie, std::shared_ptr<Labels> labels=NULL);

		virtual ~SalzbergWordStringKernel();

		/** set prior probs
		 *
		 * @param pos_prior_ positive prior
		 * @param neg_prior_ negative prior
		 */
		void set_prior_probs(float64_t pos_prior_, float64_t neg_prior_)
		{
			pos_prior=pos_prior_ ;
			neg_prior=neg_prior_ ;
			if (fabs(pos_prior+neg_prior-1)>1e-6)
				io::warn("priors don't sum to 1: {}+{}-1={}", pos_prior, neg_prior, pos_prior+neg_prior-1);
		};

		/** set prior probs from labels
		 *
		 * @param labels labels to set prior probabilites from
		 */
		void set_prior_probs_from_labels(std::shared_ptr<Labels> labels);

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r);

		/** clean up kernel */
		virtual void cleanup();

		/** return what type of kernel we are
		 *
		 * @return kernel type SALZBERG
		 */
		virtual EKernelType get_kernel_type() { return K_SALZBERG; }

		/** return the kernel's name
		 *
		 * @return name Salzberg
		 */
		virtual const char* get_name() const { return "SalzbergWordStringKernel" ; }

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
		//	float64_t compute_slow(int64_t idx_a, int64_t idx_b);

		/** compute index of given symbol at given position
		 *
		 * @param position position
		 * @param symbol symbol
		 * @return index
		 */
		inline int32_t compute_index(int32_t position, uint16_t symbol)
		{
			return position*num_symbols+symbol;
		}
	private:
		void init();

	protected:
		/** the plugin estimate */
		std::shared_ptr<PluginEstimate> estimate;

		/** mean */
		float64_t* mean;
		/** variance */
		float64_t* variance;

		/** sqrt diagonal of left-hand side */
		float64_t* sqrtdiag_lhs;
		/** sqrt diagonal of right-hand side */
		float64_t* sqrtdiag_rhs;

		/** ld mean left-hand side */
		float64_t* ld_mean_lhs;
		/** ld mean right-hand side */
		float64_t* ld_mean_rhs;

		/** number of params */
		int32_t num_params;
		/** number of symbols */
		int32_t num_symbols;
		/** sum m2 s2 */
		float64_t sum_m2_s2;
		/** positive prior */
		float64_t pos_prior;
		/** negative prior */
		float64_t neg_prior;
		/** if kernel is initialized */
		bool initialized;
};
}
#endif /* _SALZBERGWORDKERNEL_H__ */
