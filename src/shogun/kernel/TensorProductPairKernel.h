/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _TPPKKERNEL_H___
#define _TPPKKERNEL_H___

#include <shogun/lib/common.h>
#include <shogun/kernel/DotKernel.h>
#include <shogun/features/DenseFeatures.h>

namespace shogun
{
/** @brief Computes the Tensor Product Pair Kernel (TPPK).
 *
 * Formally, it computes
 *
 * \f[
 * k_{\mbox{tppk}}(({\bf a},{\bf b}), ({\bf c},{\bf d}))= k({\bf a}, {\bf c})\cdot k({\bf b}, {\bf c}) + k({\bf a},{\bf d})\cdot k({\bf b}, {\bf c})
 * \f]
 *
 * It is defined on pairs of inputs and a subkernel \f$k\f$. The subkernel has
 * to be given on initialization. The pairs are specified via indizes (ab)using
 * 2-dimensional integer features.
 *
 * Its feature space \f$\Phi_{\mbox{tppk}}\f$ is the tensor product of the
 * feature spaces of the subkernel \f$k(.,.)\f$ on its input.
 *
 * It is often used in bioinformatics, e.g., to predict protein-protein interactions.
 */
class CTensorProductPairKernel: public CDotKernel
{
	public:
		/** default constructor  */
		CTensorProductPairKernel();

		/** constructor
		 *
		 * @param size cache size
		 * @param subkernel the subkernel
		 */
		CTensorProductPairKernel(int32_t size, CKernel* subkernel);

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @param subkernel the subkernel
		 */
		CTensorProductPairKernel(CDenseFeatures<int32_t> *l, CDenseFeatures<int32_t> *r, CKernel* subkernel);

		virtual ~CTensorProductPairKernel();

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool init(CFeatures* l, CFeatures* r);

		/** return what type of kernel we are
		 *
		 * @return kernel type TPPK
		 */
		virtual EKernelType get_kernel_type() { return K_TPPK; }

		/* register the parameters
		 */
		virtual void register_params();

		/** return the kernel's name
		 *
		 * @return name TPPK
		 */
		virtual const char* get_name() const { return "TensorProductPairKernel"; }

		/** return feature class the kernel can deal with
		 *
		 * @return feature class SIMPLE
		 */
		virtual EFeatureClass get_feature_class() { return C_DENSE; }

		/** return feature type the kernel can deal with
		 *
		 * @return int32_t feature type
		 */
		virtual EFeatureType get_feature_type() { return F_INT; }

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
		/** the subkernel */
		CKernel* subkernel;
};
}
#endif /* _TPPKKERNEL_H__ */
