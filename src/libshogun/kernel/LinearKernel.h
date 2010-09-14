/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2010 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#ifndef _LINEARKERNEL_H___
#define _LINEARKERNEL_H___

#include "lib/common.h"
#include "kernel/DotKernel.h"
#include "features/DotFeatures.h"

namespace shogun
{
	class CDotFeatures;

/** @brief Computes the standard linear kernel on CDotFeatures.
 *
 * Formally, it computes
 *
 * \f[
 * k({\bf x},{\bf x'})= {\bf x}\cdot {\bf x'}
 * \f]
 */
class CLinearKernel: public CDotKernel
{
	public:
		/** constructor
		 */
		CLinearKernel();

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 */
		CLinearKernel(CDotFeatures* l, CDotFeatures* r);

		virtual ~CLinearKernel();

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
		 * @return kernel type LINEAR
		 */
		virtual EKernelType get_kernel_type() { return K_LINEAR; }

		/** return the kernel's name
		 *
		 * @return name Lineaer
		 */
		virtual const char* get_name() const { return "Linear"; }

		/** optimizable kernel, i.e. precompute normal vector and as
		 * phi(x) = x do scalar product in input space
		 *
		 * @param num_suppvec number of support vectors
		 * @param sv_idx support vector index
		 * @param alphas alphas
		 * @return if optimization was successful
		 */
		virtual bool init_optimization(
			int32_t num_suppvec, int32_t* sv_idx, float64_t* alphas);

		/** delete optimization
		 *
		 * @return if deleting was successful
		 */
		virtual bool delete_optimization();

		/** compute optimized
	 	*
	 	* @param idx index to compute
	 	* @return optimized value at given index
	 	*/
		virtual float64_t compute_optimized(int32_t idx);

		/** clear normal vector */
		virtual void clear_normal();

		/** add to normal vector
		 *
		 * @param idx where to add
		 * @param weight what to add
		 */
		virtual void add_to_normal(int32_t idx, float64_t weight);

		/** get normal
		 *
		 * @param len where length of normal vector will be stored
		 * @return normal vector
		 */
		inline const float64_t* get_normal(int32_t& len)
		{
			if (lhs && normal)
			{
				len = ((CDotFeatures*) lhs)->get_dim_feature_space();
				return normal;
			}
			else
			{
				len = 0;
				return NULL;
			}
		}

		/** get normal vector (swig compatible)
		 *
		 * @param dst_w store w in this argument
		 * @param dst_dims dimension of w
		 */
		inline void get_w(float64_t** dst_w, int32_t* dst_dims)
		{
			ASSERT(lhs && normal);
			int32_t len = ((CDotFeatures*) lhs)->get_dim_feature_space();
			ASSERT(dst_w && dst_dims);
			*dst_dims=len;
			*dst_w=(float64_t*) malloc(sizeof(float64_t)*(*dst_dims));
			ASSERT(*dst_w);
			memcpy(*dst_w, normal, sizeof(float64_t) * (*dst_dims));
		}

		/** set normal vector (swig compatible)
		 *
		 * @param src_w new w
		 * @param src_w_dim dimension of new w - must fit dim of lhs
		 */
		inline void set_w(float64_t* src_w, int32_t src_w_dim)
		{
			ASSERT(lhs && src_w_dim==((CDotFeatures*) lhs)->get_dim_feature_space());
			clear_normal();
			memcpy(normal, src_w, sizeof(float64_t) * src_w_dim);
		}

	protected:
		/** normal vector (used in case of optimized kernel) */
		float64_t* normal;
		/** length of normal vector */
		int32_t normal_length;
};
}
#endif /* _LINEARKERNEL_H__ */
