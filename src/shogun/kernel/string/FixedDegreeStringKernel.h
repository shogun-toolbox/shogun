/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _FIXEDDEGREESTRINGKERNEL_H___
#define _FIXEDDEGREESTRINGKERNEL_H___

#include <shogun/lib/common.h>
#include <shogun/kernel/string/StringKernel.h>

namespace shogun
{
/** @brief The FixedDegree String kernel takes as input two strings of same size
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
class CFixedDegreeStringKernel: public CStringKernel<char>
{
	void init();

	public:
		/** default constructor  */
		CFixedDegreeStringKernel();

		/** constructor
		 *
		 * @param size cache size
		 * @param degree the degree
		 */
		CFixedDegreeStringKernel(int32_t size, int32_t degree);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param degree the degree
		 */
		CFixedDegreeStringKernel(
			CStringFeatures<char>* l, CStringFeatures<char>* r,
			int32_t degree);

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
		virtual const char* get_name() const{ return "FixedDegreeStringKernel"; }

	protected:
		/** compute kernel function for features a and b
		 * idx_{a,b} denote the index of the feature vectors
		 * in the corresponding feature object
		 *
		 * @param idx_a index a
		 * @param idx_b index b
		 * @return computed kernel function at indices a,b
		 */
		float64_t compute(int32_t idx_a, int32_t idx_b);
		/** the degree */
		int32_t degree;
};
}
#endif /* _FIXEDDEGREESTRINGKERNEL_H___ */
