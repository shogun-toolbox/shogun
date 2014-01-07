/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009-2010 Soeren Sonnenburg
 * Copyright (C) 2009-2010 Berlin Institute of Technology
 */

#ifndef _SNPSTRINGKERNEL_H___
#define _SNPSTRINGKERNEL_H___

#include <lib/common.h>
#include <lib/memory.h>
#include <kernel/string/StringKernel.h>

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
		/** default constructor  */
		CSNPStringKernel();

		/** constructor
		 *
		 * @param size cache size
		 * @param degree degree
		 * @param win_len length of local window
		 * @param inhomogene whether inhomogeneous poly
		 */
		CSNPStringKernel(int32_t size, int32_t degree, int32_t win_len, bool inhomogene);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param degree degree
		 * @param win_len length of local window
		 * @param inhomogene whether inhomogeneous poly
		 */
		CSNPStringKernel(
			CStringFeatures<char>* l, CStringFeatures<char>* r,
			int32_t degree, int32_t win_len, bool inhomogene);

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

		/** set the base string for minor aleles
		 *
		 * @param str minor freq. string
		 */
		void set_minor_base_string(const char* str)
		{
			m_str_min=get_strdup(str);
		}

		/** set the base string for major aleles
		 *
		 * @param str major freq. string
		 */
		void set_major_base_string(const char* str)
		{
			m_str_maj=get_strdup(str);
		}

		/** get the base string for minor aleles
		 *
		 * @return minor freq. string
		 */
		char* get_minor_base_string()
		{
			return m_str_min;
		}

		/** get the base string for major aleles
		 *
		 * @return major freq. string
		 */
		char* get_major_base_string()
		{
			return m_str_maj;
		}

		/** compute the minor / major alele base strings */
		void obtain_base_strings();

		/** return the kernel's name
		 *
		 * @return name PolyMatchString
		 */
		virtual const char* get_name() const { return "SNPStringKernel"; }

		/* register the parameters
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
		/** degree */
		int32_t m_degree;
		/** window length */
		int32_t m_win_len;

		/** inhomogeneous poly kernel ? */
		bool m_inhomogene;

		/** total string length / must match length of min/maj strings and
		 * string length of each vector */
		int32_t m_str_len;

		/** allele A */
		char* m_str_min;
		/** allele B */
		char* m_str_maj;

	private:
		void init();
};
}
#endif /* _SNPSTRINGKERNEL_H___ */
