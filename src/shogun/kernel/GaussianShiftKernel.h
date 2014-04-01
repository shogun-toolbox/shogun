/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008 Gunnar Raetsch
 * Copyright (C) 2008-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _GAUSSIANSHIFTKERNEL_H___
#define _GAUSSIANSHIFTKERNEL_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/features/DenseFeatures.h>

namespace shogun
{
/** @brief An experimental kernel inspired by the
 * WeightedDegreePositionStringKernel and the Gaussian kernel.
 *
 * It is computed as
 *
 * \f[
 * k({\bf x},{\bf x'})= \exp(-\frac{||{\bf x}-{\bf x'}||^2}{\tau}) +
                        \sum_{s=1}^{S_{\mathrm{max}}/S_{\mathrm{step}}} \frac{1}{2s}
						   \exp(-\frac{||{\bf x}_{[1:|{\bf x}|-sS_{\mathrm{step}}]}-{\bf x'}_{[sS_{\mathrm{step}}:|{\bf x}|]}||^2}{\tau}) +
                        \sum_{s=1}^{S_{max}/S_{step}} \frac{1}{2s}
						   \exp(-\frac{||{\bf x}_{[sS_{\mathrm{step}}:|{\bf x}|]}-{\bf x'}_{[1:|{\bf x}|-sS_{\mathrm{step}}]}||^2}{\tau}) +
 * \f]
 *
 * where \f$\tau\f$ is the kernel width. The idea is to shift the dimensions of
 * the input vectors against eachother. \f$S_{\mathrm{step}}\f$ is the step size
 * (parameter shift_step) of the shifts and \f$S_{\mathrm{max}}\f$ (parameter
 * max_shift) is the maximal shift.
 */
class CGaussianShiftKernel: public CGaussianKernel
{
	public:
		/** default constructor  */
		CGaussianShiftKernel();

		/** constructor
		 *
		 * @param size cache size
		 * @param width width
		 * @param max_shift maximum shift
		 * @param shift_step shift step
		 */
		CGaussianShiftKernel(
			int32_t size, float64_t width, int32_t max_shift,
			int32_t shift_step);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param width width
		 * @param max_shift maximum shift
		 * @param shift_step shift step
		 * @param size cache size
		 */
		CGaussianShiftKernel(
			CDenseFeatures<float64_t>* l, CDenseFeatures<float64_t>* r,
			float64_t width, int32_t max_shift, int32_t shift_step,
			int32_t size=10);

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool init(CFeatures* l, CFeatures* r)
		{
			return CGaussianKernel::init(l,r);
		}

		virtual ~CGaussianShiftKernel();

		/** return what type of kernel we are
		 *
		 * @return kernel type GAUSSIANSHIFT
		 */
		virtual EKernelType get_kernel_type() { return K_GAUSSIANSHIFT; }

		/** return the kernel's name
		 *
		 * @return name GaussianShift
		 */
		virtual const char* get_name() const { return "GaussianShiftKernel"; }

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

	private:
		void init();

	protected:
		/** maximum shift */
		int32_t max_shift;
		/** shift step */
		int32_t shift_step;
};
}
#endif /* _GAUSSIANSHIFTKERNEL_H__ */
