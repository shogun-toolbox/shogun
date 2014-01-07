/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _POLYMATCHSTRINGKERNEL_H___
#define _POLYMATCHSTRINGKERNEL_H___

#include <lib/common.h>
#include <kernel/string/StringKernel.h>

namespace shogun
{
/** @brief The class PolyMatchStringKernel computes a variant of the polynomial
 * kernel on strings of same length.
 *
 * It is computed as
 *
 * \f[
 * k({\bf x},{\bf x'})= (\sum_{i=0}^L I(x_i=x'_i)+c)^d
 * \f]
 *
 * where I is the indicator function which evaluates to 1 if its argument is
 * true and to 0 otherwise.
 *
 * Note that additional normalisation is applied, i.e.
 * \f[
 *     k'({\bf x}, {\bf x'})=\frac{k({\bf x}, {\bf x'})}{\sqrt{k({\bf x}, {\bf x})k({\bf x'}, {\bf x'})}}
 * \f]
 */
class CPolyMatchStringKernel: public CStringKernel<char>
{
	public:
		/** default constructor  */
		CPolyMatchStringKernel();

		/** constructor
		 *
		 * @param size cache size
		 * @param degree degree
		 * @param inhomogene is inhomogeneous
		 */
		CPolyMatchStringKernel(int32_t size, int32_t degree, bool inhomogene);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param degree degree
		 * @param inhomogene is inhomogeneous
		 */
		CPolyMatchStringKernel(
			CStringFeatures<char>* l, CStringFeatures<char>* r,
			int32_t degree, bool inhomogene);

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
		virtual const char* get_name() const { return "PolyMatchStringKernel"; }

		/** enable rescaling by length of feature vector
		 *
		 * @param n true to enable
		 */
		void set_rescaling_enabled(bool n)
		{
			rescaling=n;
		}

		/** return rescaling parameter
		 *
		 * @return true if rescaling is enabled
		 */
		bool get_rescaling_enabled()
		{
			return rescaling;
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
		virtual float64_t compute(int32_t idx_a, int32_t idx_b);

	private:
		void init();

	protected:
		/** degree */
		int32_t degree;
		/** if kernel is inhomogeneous */
		bool inhomogene;
		/** true to rescale kernel with string length */
		bool rescaling;
};
}
#endif /* _POLYMATCHSTRINGKERNEL_H___ */
