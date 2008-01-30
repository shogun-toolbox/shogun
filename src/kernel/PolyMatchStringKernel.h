/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _POLYMATCHSTRINGKERNEL_H___
#define _POLYMATCHSTRINGKERNEL_H___

#include "lib/common.h"
#include "kernel/StringKernel.h"

/** kernel PolyMatchString */
class CPolyMatchStringKernel: public CStringKernel<CHAR>
{
	public:
		/** constructor
		 *
		 * @param size cache size
		 * @param degree degree
		 * @param inhomogene is inhomogeneous
		 * @param use_normalization use normalization
		 */
		CPolyMatchStringKernel(INT size, INT degree, bool inhomogene,
			bool use_normalization=true);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param degree degree
		 * @param inhomogene is inhomogeneous
		 * @param use_normalization use normalization
		 */
		CPolyMatchStringKernel(
			CStringFeatures<CHAR>* l, CStringFeatures<CHAR>* r,
			INT degree, bool inhomogene, bool use_normalization=true);

		virtual ~CPolyMatchStringKernel();

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
		 * @return kernel type POLYMATCH
		 */
		virtual EKernelType get_kernel_type()
		{
			return K_POLYMATCH;
		}

		/** return the kernel's name
		 *
		 * @return name PolyMatchString
		 */
		virtual const CHAR* get_name()
		{
			return "PolyMatchString";
		}

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
		/** degree */
		INT degree;
		/** if kernel is inhomogeneous */
		bool inhomogene;
		/** if normalization is used */
		bool use_normalization;

		/** sqrt diagonal of left-hand side */
		DREAL *sqrtdiag_lhs;
		/** sqrt diagonal of right-hand side */
		DREAL *sqrtdiag_rhs;
		/** if kernel is initialized */
		bool initialized;
};

#endif /* _POLYMATCHSTRINGKERNEL_H___ */
