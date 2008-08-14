/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _GAUSSIANSHIFTKERNEL_H___
#define _GAUSSIANSHIFTKERNEL_H___

#include "lib/common.h"
#include "kernel/GaussianKernel.h"

/** An experimental kernel inspired by the WeightedDegreePositionStringKernel
 * and the Gaussian kernel. It is computed as
 *
 * \f[
 * k({\bf x},{\bf x'})= exp(-\frac{||{\bf x}-{\bf x'}||^2}{\tau}) + \sum_{...}
 * \f]
 *
 * where \f$\tau\f$ is the kernel width.
 */
class CGaussianShiftKernel: public CGaussianKernel
{
	public:
		/** constructor
		 *
		 * @param size cache size
		 * @param width width
		 * @param max_shift maximum shift
		 * @param shift_step shift step
		 */
		CGaussianShiftKernel(INT size, double width, int max_shift,
			int shift_step);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param width width
		 * @param max_shift maximum shift
		 * @param shift_step shift step
		 * @param size cache size
		 */
		CGaussianShiftKernel(CRealFeatures* l, CRealFeatures* r,
			double width, int max_shift, int shift_step, INT size=10);

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
		virtual const CHAR* get_name() { return "GaussianShift"; }

	protected:
		/** compute kernel function for features a and b
		 * idx_{a,b} denote the index of the feature vectors
		 * in the corresponding feature object
		 *
		 * @param idx_a index a
		 * @param idx_b index b
		 * @return computed kernel function at indices a,b
		 */
		virtual DREAL compute(INT idx_a, INT idx_b);

	protected:
		/** maximum shift */
		int max_shift;
		/** shift step */
		int shift_step;
};

#endif /* _GAUSSIANSHIFTKERNEL_H__ */
