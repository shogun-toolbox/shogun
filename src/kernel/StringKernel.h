/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _STRINGKERNEL_H___
#define _STRINGKERNEL_H___

#include "kernel/Kernel.h"
#include "features/StringFeatures.h"

template <class ST> class CStringKernel : public CKernel
{
	public:
		CStringKernel(INT cachesize) : CKernel(cachesize)
		{
		}

		/** initialize your kernel
		 * where l are feature vectors to occur on left hand side
		 * and r the feature vectors to occur on right hand side
		 *
		 * when training data is supplied as both l and r do_init should
		 * be true; when testing it must be false and thus no further
		 * initialization of the preprocessor in the kernel
		 * will be done (like determining the scale factor when rescaling the kernel).
		 * instead the previous values will be used (which where hopefully obtained
		 * on training data/loaded)
		 */
		virtual bool init(CFeatures* l, CFeatures* r, bool do_init)
		{
			CKernel::init(l,r,do_init);

			ASSERT(l->get_feature_class() == C_STRING);
			ASSERT(r->get_feature_class() == C_STRING);
			ASSERT(l->get_feature_type()==this->get_feature_type());
			ASSERT(r->get_feature_type()==this->get_feature_type());

			return true;
		}

		/** return feature class the kernel can deal with
		  */
		inline virtual EFeatureClass get_feature_class() { return C_STRING; }

		/** return feature type the kernel can deal with
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

#endif

