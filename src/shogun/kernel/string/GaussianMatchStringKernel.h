/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _GAUSSIANMATCHSTRINGKERNEL_H___
#define _GAUSSIANMATCHSTRINGKERNEL_H___

#include <lib/common.h>
#include <kernel/string/StringKernel.h>

namespace shogun
{
/** @brief The class GaussianMatchStringKernel computes a variant of the Gaussian
 * kernel on strings of same length.
 *
 * It is computed as
 *
 * \f[
 * k({\bf x},{\bf x'})= e^{-\frac{\left(x-x'\right)^2}{w}}
 * \f]
 *
 *
 * Note that additional normalisation is applied, i.e.
 * \f[
 *     k'({\bf x}, {\bf x'})=\frac{k({\bf x}, {\bf x'})}{\sqrt{k({\bf x}, {\bf x})k({\bf x'}, {\bf x'})}}
 * \f]
 */
class CGaussianMatchStringKernel: public CStringKernel<char>
{
	public:
		/** default constructor  */
		CGaussianMatchStringKernel();

		/** constructor
		 *
		 * @param size cache size
		 * @param width width
		 */
		CGaussianMatchStringKernel(int32_t size, float64_t width);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param width width
		 */
		CGaussianMatchStringKernel(
			CStringFeatures<char>* l, CStringFeatures<char>* r,
			float64_t width);

		virtual ~CGaussianMatchStringKernel();

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
		 * @return kernel type POLYMATCH
		 */
		virtual EKernelType get_kernel_type()
		{
			return K_GAUSSIANMATCH;
		}

		/** return the kernel's name
		 *
		 * @return name GaussMatchStringKernel
		 */
		virtual const char* get_name() const { return "GaussianMatchStringKernel"; }
		/** register the parameters
		 */
		virtual void register_params();

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

	protected:
		/** width */
		float64_t width;
};
}
#endif /* _GAUSSIANMATCHSTRINGKERNEL_H___ */
