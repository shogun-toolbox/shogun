/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _SPARSELINEARKERNEL_H___
#define _SPARSELINEARKERNEL_H___

#include "lib/common.h"
#include "kernel/SparseKernel.h"
#include "features/SparseFeatures.h"

/** Computes the standard linear kernel on sparse real valued features
 * \f[
 * k({\bf x},{\bf x'})= \frac{1}{scale}\Phi_k({\bf x})\cdot \Phi_k({\bf x'})
 * \f]
 */
class CSparseLinearKernel: public CSparseKernel<DREAL>
{
	public:
		/** constructor
		 *
		 * @param size cache size
		 * @param scale scaling factor
		 */
		CSparseLinearKernel(INT size, DREAL scale=1.0);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param scale scaling factor
		 * @param size cache size
		 */
		CSparseLinearKernel(
			CSparseFeatures<DREAL>* l, CSparseFeatures<DREAL>* r,
			DREAL scale=1.0, INT size=10);

		virtual ~CSparseLinearKernel();

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
		 * @return kernel type SPARSELINEAR
		 */
		virtual EKernelType get_kernel_type() { return K_SPARSELINEAR; }

		/** return the kernel's name
		 *
		 * @return name FixedDegree
		 */
		virtual const CHAR* get_name() { return "SparseLinear" ; } ;

		/** optimizable kernel, i.e. precompute normal vector and as phi(x) = x
		 * do scalar product in input space
		 *
		 * @param num_suppvec number of support vectors
		 * @param sv_idx support vector index
		 * @param alphas alphas
		 * @return if optimization was successful
		 */
		virtual bool init_optimization(INT num_suppvec, INT* sv_idx,
			DREAL* alphas);

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
		virtual DREAL compute_optimized(INT idx);

		/** clear normal */
		virtual void clear_normal();

		/** add to normal
		 *
		 * @param idx where to add
		 * @param weight what to add
		 */
		virtual void add_to_normal(INT idx, DREAL weight);

		/** get normal
		 *
		 * @param len length of normal vector will be stored here
		 * @return the normal vector
		 */
		inline const double* get_normal(INT& len)
		{
			len=normal_length;
			return normal;
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

		/** initialize rescaling */
		virtual void init_rescale();

	protected:
		/** scaling factor */
		double scale;
		/** if kernel is initialized */
		bool initialized;

		/** normal vector (used in case of optimized kernel) */
		DREAL* normal;
		/** length of normal vector */
		INT normal_length;
};

#endif /* _SPARSELINEARKERNEL_H__ */
