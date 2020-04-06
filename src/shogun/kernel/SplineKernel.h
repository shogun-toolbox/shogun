/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Yuyu Zhang
 */

#ifndef _SPLINEKERNEL_H__
#define _SPLINEKERNEL_H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/kernel/DotKernel.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/machine/KernelMachine.h>

namespace shogun
{
	class KernelMachine;
	class DotFeatures;

/** @brief Computes the Spline Kernel function which is the cubic polynomial
 *
 * The formula is:
 *
 * \f[
 * k({\bf x},{\bf x'})= 1 + {\bf x} \cdot {\bf x'} +
 * {\bf x} \cdot {\bf x'} \cdot \mbox{min}(\bf x, \bf x') -
 * \frac{\bf x + \bf x'}{2} \cdot \mbox{min}(\bf x, \bf x')^2 +
 * \frac{ \mbox{min}(\bf x, \bf x')^3}{3}
 * \f]
 */
class SplineKernel: public DotKernel
{
	public:
		/** constructor
		 */
		SplineKernel();

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 */
		SplineKernel(const std::shared_ptr<DotFeatures>& l, const std::shared_ptr<DotFeatures>& r);

		~SplineKernel() override;

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
		 * @return kernel type SPLINE
		 */
		EKernelType get_kernel_type() override { return K_SPLINE; }

		/** return the kernel's name
		 *
		 * @return name Spline
		 */
		const char* get_name() const override { return "SplineKernel"; }

	protected:
		float64_t compute(int32_t idx_a, int32_t idx_b) override;
};
}
#endif /* _SPLINEKERNEL_H__ */
