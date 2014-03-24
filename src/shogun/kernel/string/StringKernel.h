/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _STRINGKERNEL_H___
#define _STRINGKERNEL_H___

#include <shogun/lib/config.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/features/StringFeatures.h>

namespace shogun
{
/** @brief Template class StringKernel, is the base class of all String Kernels.
 *
 * For a (very complex) example see e.g. CWeightedDegreeStringKernel
 *
 */
template <class ST> class CStringKernel : public CKernel
{
	public:
		/** constructor
		 *
		 * @param cachesize cache size
		 */
		CStringKernel(int32_t cachesize=0) : CKernel(cachesize) {}

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 */
		CStringKernel(CFeatures *l, CFeatures *r) : CKernel(10)
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

			ASSERT(l->get_feature_class()==C_STRING)
			ASSERT(r->get_feature_class()==C_STRING)
			ASSERT(l->get_feature_type()==this->get_feature_type())
			ASSERT(r->get_feature_type()==this->get_feature_type())

			return true;
		}

		/** return feature class the kernel can deal with
		 *
		 * @return feature class STRING
		 */
		virtual EFeatureClass get_feature_class() { return C_STRING; }

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
			return "StringKernel"; }

		/** return what type of kernel we are, e.g.
		 * Linear,Polynomial, Gaussian,...
		 *
		 * abstract base method
		 *
		 * @return kernel type
		 */
		virtual EKernelType get_kernel_type()=0;
};

template<> inline EFeatureType CStringKernel<float64_t>::get_feature_type() { return F_DREAL; }

template<> inline EFeatureType CStringKernel<uint64_t>::get_feature_type() { return F_ULONG; }

template<> inline EFeatureType CStringKernel<int32_t>::get_feature_type() { return F_INT; }

template<> inline EFeatureType CStringKernel<uint16_t>::get_feature_type() { return F_WORD; }

template<> inline EFeatureType CStringKernel<int16_t>::get_feature_type() { return F_SHORT; }

template<> inline EFeatureType CStringKernel<uint8_t>::get_feature_type() { return F_BYTE; }

template<> inline EFeatureType CStringKernel<char>::get_feature_type() { return F_CHAR; }
}
#endif /* _STRINGKERNEL_H__ */

