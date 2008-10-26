/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _GAUSSIANSHORTREALKERNEL_H___
#define _GAUSSIANSHORTREALKERNEL_H___

#include "lib/common.h"
#include "kernel/SimpleKernel.h"
#include "features/ShortRealFeatures.h"

/** The well known Gaussian kernel (swiss army knife for SVMs)
 * on dense short-real valued features is computed as
 *
 * \f[
 * k({\bf x},{\bf x'})= exp(-\frac{||{\bf x}-{\bf x'}||^2}{\tau})
 * \f]
 *
 * where \f$\tau\f$ is the kernel width.
 */
class CGaussianShortRealKernel: public CSimpleKernel<SHORTREAL>
{
	public:
		/** constructor
		 *
		 * @param size cache size
		 * @param width width
		 */
		CGaussianShortRealKernel(int32_t size, DREAL width);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param width width
		 * @param size cache size
		 */
		CGaussianShortRealKernel(CShortRealFeatures* l, CShortRealFeatures* r,
			DREAL width, int32_t size=10);

		virtual ~CGaussianShortRealKernel();

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool init(CFeatures* l, CFeatures* r);

		/** load kernel init_data
		 *
		 * @param src file to load from
		 * @return if loading was successful
		 */
		virtual bool load_init(FILE* src);

		/** save kernel init_data
		 *
		 * @param dest file to save to
		 * @return if saving was successful
		 */
		virtual bool save_init(FILE* dest);

		/** return what type of kernel we are
		 *
		 * @return kernel type GAUSSIAN
		 */
		virtual EKernelType get_kernel_type() { return K_GAUSSIAN; }

		/** return the kernel's name
		 *
		 * @return name GaussianShortReal
		 */
		virtual const char* get_name() { return "GaussianShortReal"; }

	protected:
		/** compute kernel function for features a and b
		 * idx_{a,b} denote the index of the feature vectors
		 * in the corresponding feature object
		 *
		 * @param idx_a index a
		 * @param idx_b index b
		 * @return computed kernel function at indices a,b
		 */
		virtual DREAL compute(int32_t idx_a, int32_t idx_b);

	protected:
		/** width */
		DREAL width;
};

#endif /* _GAUSSIANSHORTREALKERNEL_H__ */
