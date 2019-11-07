/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Yuyu Zhang, Sergey Lisitsyn
 */

#ifndef _HISTOGRAMWORDKERNEL_H___
#define _HISTOGRAMWORDKERNEL_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/kernel/string/StringKernel.h>
#include <shogun/classifier/PluginEstimate.h>
#include <shogun/features/StringFeatures.h>

namespace shogun
{
	class PluginEstimate;
	template <class T> class StringFeatures;
/** @brief The HistogramWordString computes the TOP kernel on inhomogeneous
 * Markov Chains. */
class HistogramWordStringKernel: public StringKernel<uint16_t>
{
	public:
		/** default constructor  */
		HistogramWordStringKernel();

		/** constructor
		 *
		 * @param size cache size
		 * @param pie plugin estimate
		 */
		HistogramWordStringKernel(int32_t size, std::shared_ptr<PluginEstimate> pie);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param pie plugin estimate
		 */
		HistogramWordStringKernel(
			const std::shared_ptr<StringFeatures<uint16_t>>& l, const std::shared_ptr<StringFeatures<uint16_t>>& r,
			std::shared_ptr<PluginEstimate> pie);

		virtual ~HistogramWordStringKernel();

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
		 * @return kernel type HISTOGRAM
		 */
		virtual EKernelType get_kernel_type() { return K_HISTOGRAM; }

		/** return the kernel's name
		 *
		 * @return name Histogram
		 */
		virtual const char* get_name() const { return "HistogramWordStringKernel" ; }

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

		/** compute index
		 *
		 * @param position position
		 * @param symbol symbol
		 * @return index at given position in given symbol
		 */
		inline int32_t compute_index(int32_t position, uint16_t symbol)
		{
			return position*num_symbols+symbol+1;
		}

	private:
		void init();

	protected:
		/** plugin estimate */
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

		/** plo left-hand side */
		float64_t* plo_lhs;
		/** plo right-hand side */
		float64_t* plo_rhs;

		/** number of parameters */
		int32_t num_params;
		/** number of parameters2 */
		int32_t num_params2;
		/** number of symbols */
		int32_t num_symbols;
		/** sum m2 s2 */
		float64_t sum_m2_s2;

		/** if kernel is initialized */
		bool initialized;
};
}
#endif /* _HISTOGRAMWORDKERNEL_H__ */
