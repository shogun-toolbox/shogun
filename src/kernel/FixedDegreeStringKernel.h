/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _FIXEDDEGREESTRINGKERNEL_H___
#define _FIXEDDEGREESTRINGKERNEL_H___

#include "lib/common.h"
#include "kernel/StringKernel.h"

/** The FixedDegree String kernel takes as input two strings of same size
 * and counts the number of matches of length d.
 *
 * \f[
 *     k({\bf x}, {\bf x'}) = \sum_{i=0}^{l-d} I({\bf x}_{i,i+1,\dots,i+d-1} = {\bf x'}_{i,i+1,\dots,i+d-1})
 * \f]
 *
 * Note that additional normalisation is applied, i.e.
 * \f[
 *     k'({\bf x}, {\bf x'})=\frac{k({\bf x}, {\bf x'})}{\sqrt{k({\bf x}, {\bf x})k({\bf x'}, {\bf x'})}}
 * \f]
 */
class CFixedDegreeStringKernel: public CStringKernel<CHAR>
{
	public:
		/** constructor
		 *
		 * @param size cache size
		 * @param degree the degree
		 */
		CFixedDegreeStringKernel(INT size, INT degree);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param degree the degree
		 */
		CFixedDegreeStringKernel(
			CStringFeatures<CHAR>* l, CStringFeatures<CHAR>* r,
			INT degree);

		virtual ~CFixedDegreeStringKernel();

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
		bool load_init(FILE* src);

		/** save kernel init_data
		 *
		 * @param dest file to save to
		 * @return if saving was successful
		 */
		bool save_init(FILE* dest);

		/** return what type of kernel we are
		 *
		 * @return kernel type FIXEDDEGREE
		 */
		virtual EKernelType get_kernel_type()
		{
			return K_FIXEDDEGREE;
		}

		/** return the kernel's name
		 *
		 * @return name FixedDegree
		 */
		virtual const CHAR* get_name()
		{
			return "FixedDegree";
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
		DREAL compute(INT idx_a, INT idx_b);
		/** the degree */
		INT degree;
		/** sqrt diagonal of left-hand side */
		DREAL *sqrtdiag_lhs;
		/** sqrt diagonal of right-hand side */
		DREAL *sqrtdiag_rhs;
		/** if kernel is initialized */
		bool initialized;
};

#endif /* _FIXEDDEGREESTRINGKERNEL_H___ */
