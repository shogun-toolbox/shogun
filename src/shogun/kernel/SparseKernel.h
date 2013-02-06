/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _SPARSEKERNEL_H___
#define _SPARSEKERNEL_H___

#include <shogun/kernel/Kernel.h>
#include <shogun/features/SparseFeatures.h>

namespace shogun
{
/** @brief Template class SparseKernel, is the base class of kernels working on sparse
 * features.
 *
 * See e.g. the CSparseGaussianKernel for an example.
 */
template <class ST> class CSparseKernel : public CKernel
{
	public:
		/** constructor
		 *
		 * @param cachesize cache size
		 */
		CSparseKernel(int32_t cachesize) : CKernel(cachesize) {}

		/** constructor
		 *
		 * @param l features for left-hand side
		 * @param r features for right-hand side
		 */
		CSparseKernel(CFeatures* l, CFeatures* r) : CKernel(10)
		{
			init(l, r);
		}

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool init(CFeatures* l, CFeatures* r)
		{
			CKernel::init(l,r);

			ASSERT(l->get_feature_class()==C_SPARSE)
			ASSERT(r->get_feature_class()==C_SPARSE)
			ASSERT(l->get_feature_type()==this->get_feature_type())
			ASSERT(r->get_feature_type()==this->get_feature_type())

			if (((CSparseFeatures<ST>*) lhs)->get_num_features() != ((CSparseFeatures<ST>*) rhs)->get_num_features())
			{
				SG_ERROR("train or test features #dimension mismatch (l:%d vs. r:%d)\n",
						((CSparseFeatures<ST>*) lhs)->get_num_features(),((CSparseFeatures<ST>*)rhs)->get_num_features());
			}
			return true;
		}

		/** return feature class the kernel can deal with
		 *
		 * @return feature class SPARSE
		 */
		virtual EFeatureClass get_feature_class() { return C_SPARSE; }

		/** return feature type the kernel can deal with
		 *
		 * @return templated feature type
		 */
		virtual EFeatureType get_feature_type();

		/** Returns the name of the SGSerializable instance.  It MUST BE
		 *  the CLASS NAME without the prefixed `C'.
		 *
		 *  @return name of the SGSerializable
		 */
		virtual const char* get_name() const {
			return "SparseKernel"; }

		/** return what type of kernel we are, e.g.
		 * Linear,Polynomial, Gaussian,...
		 *
		 * abstract base method
		 *
		 * @return kernel type
		 */
		virtual EKernelType get_kernel_type()=0;
};

template<> inline EFeatureType CSparseKernel<float64_t>::get_feature_type() { return F_DREAL; }

template<> inline EFeatureType CSparseKernel<uint64_t>::get_feature_type() { return F_ULONG; }

template<> inline EFeatureType CSparseKernel<int32_t>::get_feature_type() { return F_INT; }

template<> inline EFeatureType CSparseKernel<uint16_t>::get_feature_type() { return F_WORD; }

template<> inline EFeatureType CSparseKernel<int16_t>::get_feature_type() { return F_SHORT; }

template<> inline EFeatureType CSparseKernel<uint8_t>::get_feature_type() { return F_BYTE; }

template<> inline EFeatureType CSparseKernel<char>::get_feature_type() { return F_CHAR; }
}
#endif /* _SPARSEKERNEL_H__ */
