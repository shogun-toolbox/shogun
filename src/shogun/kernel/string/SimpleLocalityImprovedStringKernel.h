/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Gunnar Raetsch
 * Written (W) 2013 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 * Copyright (C) 2013 Soeren Sonnenburg
 */

#ifndef _SIMPLELOCALITYIMPROVEDSTRINGKERNEL_H___
#define _SIMPLELOCALITYIMPROVEDSTRINGKERNEL_H___

#include <shogun/lib/common.h>
#include <shogun/kernel/string/StringKernel.h>

namespace shogun
{
/** @brief SimpleLocalityImprovedString kernel, is a ``simplified'' and better
 * performing version of the Locality improved kernel.
 *
 * It can be defined as
 *
 * \f[
 * K({\bf x},{\bf x'})=\left(\sum_{i=0}^{T-1}\left(\sum_{j=-l}^{+l}w_jI_{i+j}({\bf x},{\bf x'})\right)^{d_1}\right)^{d_2},
 * \f]
 * where
 * \f$ I_i({\bf x},{\bf x'})=1\f$ if \f$x_i=x'_i\f$ and 0 otherwise.
 */
class CSimpleLocalityImprovedStringKernel: public CStringKernel<char>
{
	public:
		/** default constructor  */
		CSimpleLocalityImprovedStringKernel();

		/** constructor
		 *
		 * @param size cache size
		 * @param length length
		 * @param inner_degree inner degree
		 * @param outer_degree outer degree
		 */
		CSimpleLocalityImprovedStringKernel(int32_t size, int32_t length,
			int32_t inner_degree, int32_t outer_degree);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param length length
		 * @param inner_degree inner degree
		 * @param outer_degree outer degree
		 */
		CSimpleLocalityImprovedStringKernel(
			CStringFeatures<char>* l, CStringFeatures<char>* r,
			int32_t length, int32_t inner_degree, int32_t outer_degree);

		virtual ~CSimpleLocalityImprovedStringKernel();

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool init(CFeatures *l, CFeatures *r);

		/** clean up kernel */
		virtual void cleanup();

		/** return what type of kernel we are
		 *
		 * @return kernel type SIMPLELOCALITYIMPROVED
		 */
		virtual EKernelType get_kernel_type()
		{
			return K_SIMPLELOCALITYIMPROVED;
		}

		/** return the kernel's name
		 *
		 * @return name SimpleLocalityImprovedStringKernel
		 */
		virtual const char* get_name() const
		{
			return "SimpleLocalityImprovedStringKernel";
		}

	private:
		/** dot pyr
		 *
		 * @param x1 x1
		 * @param x2 x2
		 * @param NOF_NTS NOF NTS
		 * @param NTWIDTH NT width
		 * @param DEGREE1 degree 1
		 * @param DEGREE2 degree 2
		 * @param pyra pyramid
		 * @return dot product of pyramid (?)
		 */
		float64_t dot_pyr (const char* const x1, const char* const x2,
				const int32_t NOF_NTS, const int32_t NTWIDTH,
				const int32_t DEGREE1, const int32_t DEGREE2, float64_t *pyra);

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

	private:
		void init();

	protected:
		/** length */
		int32_t length;
		/** inner degree */
		int32_t inner_degree;
		/** outer degree */
		int32_t outer_degree;

		/** pyramid weights */
		SGVector<float64_t> pyramid_weights;
};
}
#endif /* _SIMPLELOCALITYIMPROVEDSTRINGKERNEL_H___ */
