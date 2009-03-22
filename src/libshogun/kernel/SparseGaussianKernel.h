/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _SPARSEGAUSSIANKERNEL_H___
#define _SPARSEGAUSSIANKERNEL_H___

#include "lib/common.h"
#include "kernel/SparseKernel.h"
#include "features/SparseFeatures.h"

/** @brief The well known Gaussian kernel (swiss army knife for SVMs)
 * on sparse real valued features.
 *
 * It is computed as
 *
 * \f[
 * k({\bf x},{\bf x'})= exp(-\frac{||{\bf x}-{\bf x'}||^2}{\tau})
 * \f]
 *
 * where \f$\tau\f$ is the kernel width.
 */
class CSparseGaussianKernel: public CSparseKernel<float64_t>
{
	public:
		/** constructor
		 *
		 * @param size cache size
		 * @param width width
		 */
		CSparseGaussianKernel(int32_t size, float64_t width);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param width width
		 */
		CSparseGaussianKernel(
			CSparseFeatures<float64_t>* l, CSparseFeatures<float64_t>* r,
			float64_t width);

		virtual ~CSparseGaussianKernel();

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
		 * @return kernel type SPARSEGAUSSIAN
		 */
		virtual EKernelType get_kernel_type() { return K_SPARSEGAUSSIAN; }

		/** return feature type the kernel can deal with
		 *
		 * @return feature type DREAL
		 */
		inline virtual EFeatureType get_feature_type() { return F_DREAL; }

		/** return the kernel's name
		 *
		 * @return name SparseGaussian
		 */
		virtual const char* get_name() const { return "SparseGaussian"; }

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
		/** squared left-hand side */
		float64_t* sq_lhs;
		/** squared right-hand side */
		float64_t* sq_rhs;
};

#endif /* _SPARSEGAUSSIANKERNEL_H__ */
