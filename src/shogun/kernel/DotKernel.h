/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#ifndef _DOTKERNEL_H___
#define _DOTKERNEL_H___

#include <shogun/kernel/Kernel.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/io/SGIO.h>

namespace shogun
{
/** @brief Template class DotKernel is the base class for kernels working on
 * DotFeatures.
 *
 * CDotFeatures are features supporting operations like dot product, dot product
 * with a dense vector and addition to a dense vector. Therefore several dot
 * product based kernels derive from this class (cf., e.g., CLinearKernel)
 *
 * \sa CDotFeatures
 */
class CDotKernel : public CKernel
{
	public:
		/** default constructor
		 *
		 */
		CDotKernel() : CKernel() {}

		/** constructor
		 *
		 * @param cachesize cache size
		 */
		CDotKernel(int32_t cachesize) : CKernel(cachesize) {}

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 */
		CDotKernel(CFeatures* l, CFeatures* r) : CKernel(10)
		{
			init(l, r);
		}

		/** initialize kernel
		 *  e.g. setup lhs/rhs of kernel, precompute normalization
		 *  constants etc.
		 *  make sure to check that your kernel can deal with the
		 *  supplied features (!)
		 *
		 *  @param l features for left-hand side
		 *  @param r features for right-hand side
		 *  @return if init was successful
		 */
		virtual bool init(CFeatures* l, CFeatures* r)
		{
			CKernel::init(l,r);

			ASSERT(l->has_property(FP_DOT))
			ASSERT(r->has_property(FP_DOT))
			ASSERT(l->get_feature_type() == r->get_feature_type())
			ASSERT(l->get_feature_class() == r->get_feature_class())

			if ( ((CDotFeatures*) l)->get_dim_feature_space() != ((CDotFeatures*) r)->get_dim_feature_space() )
			{
				SG_ERROR("train or test features #dimension mismatch (l:%d vs. r:%d)\n",
						((CDotFeatures*) l)->get_dim_feature_space(),((CDotFeatures*) r)->get_dim_feature_space());
			}
			return true;
		}

		/** return feature class the kernel can deal with
		 *
		 * dot kernel returns unknown since features can be based on anything
		 *
		 * @return feature class ANY
		 */
		virtual EFeatureClass get_feature_class() { return C_ANY; }

		/** return feature type the kernel can deal with
		 *
		 * dot kernel returns unknown since features can be based on anything
		 *
		 * @return ANY feature type
		 */
		virtual EFeatureType get_feature_type() { return F_ANY; }

		/** Returns the name of the SGSerializable instance.  It MUST BE
		 *  the CLASS NAME without the prefixed `C'.
		 *
		 * @return name of the SGSerializable
		 */
		virtual const char* get_name() const { return "DotKernel"; }

		/** return what type of kernel we are, e.g.
		 * Linear,Polynomial, Gaussian,...
		 *
		 * abstract base method
		 *
		 * @return kernel type
		 */
		virtual EKernelType get_kernel_type()=0 ;

	protected:
		/** compute kernel function for features a and b
		 * idx_{a,b} denote the index of the feature vectors
		 * in the corresponding feature object
		 *
		 * @param idx_a index a
		 * @param idx_b index b
		 * @return computed kernel function at indices a,b
		 */
		virtual float64_t compute(int32_t idx_a, int32_t idx_b)
		{
			return ((CDotFeatures*) lhs)->dot(idx_a, ((CDotFeatures*) rhs), idx_b);
		}
};
}
#endif /* _DOTKERNEL_H__ */

