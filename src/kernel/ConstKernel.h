/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _CONSTKERNEL_H___
#define _CONSTKERNEL_H___

#include "lib/Mathematics.h"
#include "lib/common.h"
#include "kernel/Kernel.h"
#include "features/Features.h"

/** Constant Kernel
 *
 * A ``kernel'' that simply returns a single constant, i.e.
 * \f$k({\bf x}, {\bf x'})= c\f$
 *
 */
class CConstKernel: public CKernel
{
	public:
		/** constructor
		 *
		 * @param c constant c
		 */
		CConstKernel(DREAL c);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param c constant c
		 */
		CConstKernel(CFeatures* l, CFeatures *r, DREAL c);

		virtual ~CConstKernel();

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
		 * @return kernel type CONST
		 */
		inline virtual EKernelType get_kernel_type() { return K_CONST; }

		/** return feature type the kernel can deal with
		 *
		 * @return feature type ANY
		 */
		inline virtual EFeatureType get_feature_type()
		{
			return F_ANY;
		}

		/** return feature class the kernel can deal with
		 *
		 * @return feature class ANY
		 */
		inline virtual EFeatureClass get_feature_class()
		{
			return C_ANY;
		}

		/** return the kernel's name
		 *
		 * @return name Const
		 */
		virtual const char* get_name() { return "Const"; }

	protected:
		/** compute kernel function for features a and b
		 *
		 * @param row dummy row
		 * @param col dummy col
		 * @return computed kernel function (const value)
		 */
		inline virtual DREAL compute(INT row, INT col)
		{
			return const_value;
		}

	protected:
		/** const value */
		DREAL const_value;
};

#endif /* _CONSTKERNEL_H__ */
