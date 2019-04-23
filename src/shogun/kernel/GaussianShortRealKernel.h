/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Yuyu Zhang, Sergey Lisitsyn
 */

#ifndef _GAUSSIANSHORTREALKERNEL_H___
#define _GAUSSIANSHORTREALKERNEL_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/kernel/DotKernel.h>
#include <shogun/features/DenseFeatures.h>

namespace shogun
{
/** @brief The well known Gaussian kernel (swiss army knife for SVMs)
 * on dense short-real valued features.
 *
 * It is computed as
 *
 * \f[
 * k({\bf x},{\bf x'})= exp(-\frac{||{\bf x}-{\bf x'}||^2}{\tau})
 * \f]
 *
 * where \f$\tau\f$ is the kernel width.
 */
class GaussianShortRealKernel: public DotKernel
{
	public:
		/** default constructor  */
		GaussianShortRealKernel();

		/** constructor
		 *
		 * @param size cache size
		 * @param width width
		 */
		GaussianShortRealKernel(int32_t size, float64_t width);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param width width
		 * @param size cache size
		 */
		GaussianShortRealKernel(std::shared_ptr<DenseFeatures<float32_t>> l, std::shared_ptr<DenseFeatures<float32_t>> r,
			float64_t width, int32_t size=10);

		virtual ~GaussianShortRealKernel();

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r);

		/** return what type of kernel we are
		 *
		 * @return kernel type GAUSSIAN
		 */
		virtual EKernelType get_kernel_type() { return K_GAUSSIAN; }

		/** return the kernel's name
		 *
		 * @return name GaussianShortReal
		 */
		virtual const char* get_name() const { return "GaussianShortRealKernel"; }
		/** register the parameters
		 */
		virtual void register_params();

	protected:
		/** compute kernel function for features a and b
		 * idx_{a,b} denote the index of the feature vectors
		 * in the corresponding feature object
		 *
		 * @param idx_a index a
		 * @param idx_b index b
		 * @return computed kernel function at indices a,b
		 */
		virtual float64_t compute(int32_t idx_a, int32_t idx_b);

	protected:
		/** width */
		float64_t width;
};
}
#endif /* _GAUSSIANSHORTREALKERNEL_H__ */
