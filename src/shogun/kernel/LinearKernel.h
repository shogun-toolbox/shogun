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

#include <shogun/lib/common.h>
#include <shogun/kernel/DotKernel.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/machine/KernelMachine.h>

namespace shogun
{
	class CKernelMachine;
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
		virtual const char* get_name() const { return "LinearKernel"; }

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

		/** init optimization
		 * @param km
		 */
		virtual bool init_optimization(CKernelMachine* km);

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

		virtual void clear_normal()
		{
			normal = SGVector<float64_t>(((CDotFeatures*)lhs)->get_dim_feature_space());
			normal.zero();
			set_is_initialized(true);
		}

		/** add to normal vector
		 *
		 * @param idx where to add
		 * @param weight what to add
		 */
		virtual void add_to_normal(int32_t idx, float64_t weight);

		/** get normal vector
		 *
		 * @return normal vector
		 */
		SGVector<float64_t> get_w() const
		{
			ASSERT(lhs)
			return normal;
		}

		/** set normal vector
		 *
		 * @param w new normal
		 */
		void set_w(SGVector<float64_t> w)
		{
			ASSERT(lhs && w.size()==((CDotFeatures*) lhs)->get_dim_feature_space())
			this->normal = w;
		}

	protected:
		/** normal vector (used in case of optimized kernel) */
		SGVector<float64_t> normal;
};
}
#endif /* _LINEARKERNEL_H__ */
