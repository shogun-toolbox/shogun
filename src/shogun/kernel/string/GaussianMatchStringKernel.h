/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Yuyu Zhang, Sergey Lisitsyn
 */

#ifndef _GAUSSIANMATCHSTRINGKERNEL_H___
#define _GAUSSIANMATCHSTRINGKERNEL_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/kernel/string/StringKernel.h>

namespace shogun
{
/** @brief The class GaussianMatchStringKernel computes a variant of the Gaussian
 * kernel on strings of same length.
 *
 * It is computed as
 *
 * \f[
 * k({\bf x},{\bf x'})= e^{-\frac{\left(x-x'\right)^2}{w}}
 * \f]
 *
 *
 * Note that additional normalisation is applied, i.e.
 * \f[
 *     k'({\bf x}, {\bf x'})=\frac{k({\bf x}, {\bf x'})}{\sqrt{k({\bf x}, {\bf x})k({\bf x'}, {\bf x'})}}
 * \f]
 */
class GaussianMatchStringKernel: public StringKernel<char>
{
	public:
		/** default constructor  */
		GaussianMatchStringKernel();

		/** constructor
		 *
		 * @param size cache size
		 * @param width width
		 */
		GaussianMatchStringKernel(int32_t size, float64_t width);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param width width
		 */
		GaussianMatchStringKernel(
			const std::shared_ptr<StringFeatures<char>>& l, const std::shared_ptr<StringFeatures<char>>& r,
			float64_t width);

		virtual ~GaussianMatchStringKernel();

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
		 * @return kernel type POLYMATCH
		 */
		virtual EKernelType get_kernel_type()
		{
			return K_GAUSSIANMATCH;
		}

		/** return the kernel's name
		 *
		 * @return name GaussMatchStringKernel
		 */
		virtual const char* get_name() const { return "GaussianMatchStringKernel"; }
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
#endif /* _GAUSSIANMATCHSTRINGKERNEL_H___ */
