/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _CONSTKERNEL_H___
#define _CONSTKERNEL_H___

#include <shogun/lib/config.h>

#include <shogun/mathematics/Math.h>
#include <shogun/lib/common.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/features/Features.h>

namespace shogun
{
/** @brief The Constant Kernel returns a constant for all elements.
 *
 * A ``kernel'' that simply returns a single constant, i.e.
 * \f$k({\bf x}, {\bf x'})= c\f$
 *
 */
class CConstKernel: public CKernel
{
	public:
		/** default constructor  */
		CConstKernel();

		/** constructor
		 *
		 * @param c constant c
		 */
		CConstKernel(float64_t c);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param c constant c
		 */
		CConstKernel(CFeatures* l, CFeatures *r, float64_t c);

		virtual ~CConstKernel();

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool init(CFeatures* l, CFeatures* r);

		/** return what type of kernel we are
		 *
		 * @return kernel type CONST
		 */
		virtual EKernelType get_kernel_type() { return K_CONST; }

		/** return feature type the kernel can deal with
		 *
		 * @return feature type ANY
		 */
		virtual EFeatureType get_feature_type()
		{
			return F_ANY;
		}

		/** return feature class the kernel can deal with
		 *
		 * @return feature class ANY
		 */
		virtual EFeatureClass get_feature_class()
		{
			return C_ANY;
		}

		/** return the kernel's name
		 *
		 * @return name Const
		 */
		virtual const char* get_name() const { return "ConstKernel"; }

	protected:
		/** compute kernel function for features a and b
		 *
		 * @param row dummy row
		 * @param col dummy col
		 * @return computed kernel function (const value)
		 */
		virtual float64_t compute(int32_t row, int32_t col)
		{
			return const_value;
		}

	private:
		void init();

	protected:
		/** const value */
		float64_t const_value;
};
}
#endif /* _CONSTKERNEL_H__ */
