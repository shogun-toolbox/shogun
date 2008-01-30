/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _STRINGKERNEL_H___
#define _STRINGKERNEL_H___

#include "kernel/Kernel.h"
#include "features/StringFeatures.h"

/** template class StringKernel */
template <class ST> class CStringKernel : public CKernel
{
	public:
		/** constructor
		 *
		 * @param cachesize cache size
		 */
		CStringKernel(INT cachesize) : CKernel(cachesize) {}

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

			ASSERT(l->get_feature_class() == C_STRING);
			ASSERT(r->get_feature_class() == C_STRING);
			ASSERT(l->get_feature_type()==this->get_feature_type());
			ASSERT(r->get_feature_type()==this->get_feature_type());

			return true;
		}

		/** return feature class the kernel can deal with
		 *
		 * @return feature class STRING
		 */
		inline virtual EFeatureClass get_feature_class() { return C_STRING; }

		/** return feature type the kernel can deal with
		 *
		 * @return templated feature type
		 */
		virtual EFeatureType get_feature_type();
};

template<> inline EFeatureType CStringKernel<DREAL>::get_feature_type() { return F_DREAL; }

template<> inline EFeatureType CStringKernel<ULONG>::get_feature_type() { return F_ULONG; }

template<> inline EFeatureType CStringKernel<INT>::get_feature_type() { return F_INT; }

template<> inline EFeatureType CStringKernel<WORD>::get_feature_type() { return F_WORD; }

template<> inline EFeatureType CStringKernel<SHORT>::get_feature_type() { return F_SHORT; }

template<> inline EFeatureType CStringKernel<BYTE>::get_feature_type() { return F_BYTE; }

template<> inline EFeatureType CStringKernel<CHAR>::get_feature_type() { return F_CHAR; }

#endif /* _STRINGKERNEL_H__ */

