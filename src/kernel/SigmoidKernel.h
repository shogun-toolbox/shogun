/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _SIGMOIDKERNEL_H___
#define _SIGMOIDKERNEL_H___

#include "lib/common.h"
#include "kernel/SimpleKernel.h"
#include "features/RealFeatures.h"

/** kernel Sigmoid */
class CSigmoidKernel: public CSimpleKernel<DREAL>
{
	public:
		/** constructor
		 *
		 * @param size cache size
		 * @param gamma gamma
		 * @param coef0 coefficient 0
		 */
		CSigmoidKernel(INT size, DREAL gamma, DREAL coef0);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param size cache size
		 * @param gamma gamma
		 * @param coef0 coefficient 0
		 */
		CSigmoidKernel(CRealFeatures* l, CRealFeatures* r, INT size,
			DREAL gamma, DREAL coef0);

		virtual ~CSigmoidKernel();

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool init(CFeatures* l, CFeatures* r);

		/** clean up kernel */
		virtual void cleanup();

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
		 * @return kernel type SIGMOID
		 */
		virtual EKernelType get_kernel_type() { return K_SIGMOID; }

		/** return the kernel's name
		 *
		 * @return name Sigmoid
		 */
		virtual const CHAR* get_name() { return "Sigmoid" ; } ;

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
		/** gamma */
		double gamma;
		/** coefficient 0 */
		double coef0;
};

#endif /* _SIGMOIDKERNEL_H__ */
