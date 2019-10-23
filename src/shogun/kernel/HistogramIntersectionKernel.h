/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn, Yuyu Zhang
 */

#ifndef _HISTOGRAMINTERSECTIONKERNEL_H___
#define _HISTOGRAMINTERSECTIONKERNEL_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/kernel/DotKernel.h>
#include <shogun/features/DenseFeatures.h>

namespace shogun
{
/** @brief The HistogramIntersection kernel operating on realvalued vectors computes
 * the histogram intersection distance between sets of histograms.
 * Note: the current implementation assumes positive values for the histograms,
 * and input vectors should sum to 1.
 *
 * It is defined as
 * \f[
 * k({\bf x},{\bf x'})= \sum_{i=0}^{l} \mbox{min}(x^{\beta}_i, x'^{\beta}_i)
 * \f]
 * with \f$\beta=1\f$ by default
 * */
class HistogramIntersectionKernel: public DotKernel
{
	public:
		/** default constructor  */
		HistogramIntersectionKernel();

		/** constructor
		 *
		 * @param size cache size
		 */
		HistogramIntersectionKernel(int32_t size);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param beta kernel parameter
		 * @param size cache size
		 */
		HistogramIntersectionKernel(
			const std::shared_ptr<DenseFeatures<float64_t>>& l, const std::shared_ptr<DenseFeatures<float64_t>>& r,
			float64_t beta=1.0, int32_t size=10);

		virtual ~HistogramIntersectionKernel();

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r);
		/* register the parameters
		 */
		virtual void register_params();

		/** return what type of kernel we are
		 *
		 * @return kernel type HISTOGRAMINTERSECTION
		 */
		virtual EKernelType get_kernel_type() { return K_HISTOGRAMINTERSECTION; }

		/** return the kernel's name
		 *
		 * @return name HistogramIntersectionKernel
		 */
		virtual const char* get_name() const { return "HistogramIntersectionKernel"; }

		/** getter for beta parameter
		 * @return beta value
		 */
		inline float64_t get_beta() { return m_beta; }

		/** setter for beta parameter
		 *  @param beta beta value
		 */
		inline void set_beta(float64_t beta) { m_beta = beta; }

	protected:

		/// beta parameter
		float64_t m_beta;

		/** compute kernel function for features a and b
		 * idx_{a,b} denote the index of the feature vectors
		 * in the corresponding feature object
		 *
		 * @param idx_a index a
		 * @param idx_b index b
		 * @return computed kernel function at indices a,b
		 */
		virtual float64_t compute(int32_t idx_a, int32_t idx_b);

};
}
#endif /* _HISTOGRAMINTERSECTIONKERNEL_H__ */
