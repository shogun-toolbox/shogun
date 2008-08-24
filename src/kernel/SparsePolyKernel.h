/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _SPARSEPOLYKERNEL_H___
#define _SPARSEPOLYKERNEL_H___

#include "lib/common.h"
#include "kernel/SparseKernel.h"
#include "features/SparseFeatures.h"

/** Computes the standard polynomial kernel on sparse real valued features
 * \f[
 * k({\bf x},{\bf x'})= ({\bf x}\cdot {\bf x'}+c)^d
 * \f]
 *
 * Note that additional normalisation is applied, i.e.
 * \f[
 *     k'({\bf x}, {\bf x'})=\frac{k({\bf x}, {\bf x'})}{\sqrt{k({\bf x}, {\bf x})k({\bf x'}, {\bf x'})}}
 * \f]
 */
class CSparsePolyKernel: public CSparseKernel<DREAL>
{
	public:
		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param size cache size
		 * @param d degree
		 * @param inhom is inhomogeneous
		 */
		CSparsePolyKernel(
			CSparseFeatures<DREAL>* l, CSparseFeatures<DREAL>* r,
			INT size, INT d, bool inhom);

		/** constructor
		 *
		 * @param size cache size
		 * @param degree degree
		 * @param inhomogene is inhomogeneous
		 */
		CSparsePolyKernel(INT size, INT degree, bool inhomogene=true);

		virtual ~CSparsePolyKernel();

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

		/** return feature type the kernel can deal with
		 *
		 * @return feature type DREAL
		 */
		inline virtual EFeatureType get_feature_type() { return F_DREAL; }

		/** return what type of kernel we are
		 *
		 * @return kernel type POLY
		 */
		virtual EKernelType get_kernel_type() { return K_POLY; }

		/** return the kernel's name
		 *
		 * @return name SparsePoly
		 */
		virtual const CHAR* get_name() { return "SparsePoly"; }

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

	protected:
		/** degree */
		INT degree;
		/** if kernel is inhomogeneous */
		bool inhomogene;
};

#endif /* _SPARSEPOLYKERNEL_H__ */
