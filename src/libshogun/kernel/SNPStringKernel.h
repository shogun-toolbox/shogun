/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Berlin Institute of Technology
 */

#ifndef _SNPSTRINGKERNEL_H___
#define _SNPSTRINGKERNEL_H___

#include "lib/common.h"
#include "kernel/StringKernel.h"

namespace shogun
{
/** @brief The class SNPStringKernel computes a variant of the polynomial
 * kernel on strings of same length.
 *
 * It is computed as FIXME
 *
 * \f[
 * k({\bf x},{\bf x'})= (\sum_{i=0}^{L-1} I(x_i=x'_i)+c)^d
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
class CSNPStringKernel: public CStringKernel<char>
{
	public:
		/** constructor
		 *
		 * @param size cache size
		 * @param degree degree
		 */
		CSNPStringKernel(int32_t size, int32_t degree);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param degree degree
		 */
		CSNPStringKernel(
			CStringFeatures<char>* l, CStringFeatures<char>* r, int32_t degree);

		virtual ~CSNPStringKernel();

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

		void set_minor_base_string(const char* str)
		{
			str_min=strdup(str);
		}

		void set_major_base_string(const char* str)
		{
			str_maj=strdup(str);
		}

		char* get_minor_base_string()
		{
			return str_min;
		}

		char* get_major_base_string()
		{
			return str_maj;
		}

		void obtain_base_strings();

		/** return the kernel's name
		 *
		 * @return name PolyMatchString
		 */
		virtual const char* get_name() const { return "PolyMatchString"; }

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
		/** degree */
		int32_t m_degree;

		int32_t str_len;
		char* str_min;
		char* str_maj;
};
}
#endif /* _SNPSTRINGKERNEL_H___ */
