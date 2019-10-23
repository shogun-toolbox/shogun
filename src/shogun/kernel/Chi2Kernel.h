/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Heiko Strathmann, Yuyu Zhang, Sergey Lisitsyn
 */

#ifndef _CHI2KERNEL_H___
#define _CHI2KERNEL_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/kernel/DotKernel.h>
#include <shogun/features/Features.h>
#include <shogun/features/DenseFeatures.h>

namespace shogun
{
/** @brief The Chi2 kernel operating on realvalued vectors computes
 * the chi-squared distance between sets of histograms.
 *
 * It is a very useful distance in image recognition (used to detect objects).
 *
 * It is defined as
 * \f[
 * k({\bf x},{\bf x'})= e^{-\frac{1}{width} \sum_{i=0}^{l}\frac{(x_i-x'_i)^2}{(x_i+x'_i)}}
 * \f]
 *
 * */
class Chi2Kernel: public DotKernel
{
	void init();

	public:
		/** default constructor  */
		Chi2Kernel();

		/** constructor
		 *
		 * @param size cache size
		 * @param width width
		 */
		Chi2Kernel(int32_t size, float64_t width);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param width width
		 * @param size cache size
		 */
		Chi2Kernel(const std::shared_ptr<DenseFeatures<float64_t>>& l, const std::shared_ptr<DenseFeatures<float64_t>>& r,
				float64_t width, int32_t size=10);

		virtual ~Chi2Kernel();

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool init(std::shared_ptr<Features> l, std::shared_ptr<Features> r);

		/** @return width of the kernel */
		virtual float64_t get_width();

		/** @param kernel is casted to CChi2Kernel, error if not possible
		 * is SG_REF'ed
		 * @return casted GaussianKernel object
		 */
		static std::shared_ptr<Chi2Kernel> obtain_from_generic(const std::shared_ptr<Kernel>& kernel);

		/** return what type of kernel we are
		 *
		 * @return kernel type CHI2
		 */
		virtual EKernelType get_kernel_type() { return K_CHI2; }

		/** return feature class the kernel can deal with
		 *
		 * @return feature class SIMPLE
		 */
		virtual EFeatureClass get_feature_class() { return C_DENSE; }

		/** return feature type the kernel can deal with
		 *
		 * @return float64_t feature type
		 */
		virtual EFeatureType get_feature_type() { return F_DREAL; }

		/** return the kernel's name
		 *
		 * @return name Chi2
		 */
		virtual const char* get_name() const { return "Chi2Kernel"; }

		/** set width
		 *
		 *@param w width int32_t(greater than 0)
		 */
		void set_width(int32_t w);

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
#endif /* _CHI2KERNEL_H__ */
