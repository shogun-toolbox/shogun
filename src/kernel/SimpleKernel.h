/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _SIMPLEKERNEL_H___
#define _SIMPLEKERNEL_H___

#include "kernel/Kernel.h"
#include "features/SimpleFeatures.h"
#include "lib/io.h"

template <class ST> class CSimpleKernel : public CKernel
{
	public:
		CSimpleKernel(INT cachesize) : CKernel(cachesize)
		{
		}

		/** initialize your kernel
		 * where l are feature vectors to occur on left hand side
		 * and r the feature vectors to occur on right hand side
		 */
		virtual bool init(CFeatures* l, CFeatures* r)
		{
			CKernel::init(l,r);

			ASSERT(l->get_feature_class() == C_SIMPLE);
			ASSERT(r->get_feature_class() == C_SIMPLE);
			ASSERT(l->get_feature_type()==this->get_feature_type());
			ASSERT(r->get_feature_type()==this->get_feature_type());

			if ( ((CSimpleFeatures<ST>*) l)->get_num_features() != ((CSimpleFeatures<ST>*) r)->get_num_features() )
			{  
				SG_ERROR( "train or test features #dimension mismatch (l:%d vs. r:%d)\n",
						((CSimpleFeatures<ST>*) l)->get_num_features(),((CSimpleFeatures<ST>*) r)->get_num_features());
			}
			return true;
		}

		/** return feature class the kernel can deal with
		  */
		inline virtual EFeatureClass get_feature_class() { return C_SIMPLE; }

		/** return feature type the kernel can deal with
		  */
		inline virtual EFeatureType get_feature_type();
};


template<> inline EFeatureType CSimpleKernel<DREAL>::get_feature_type() { return F_DREAL; }

template<> inline EFeatureType CSimpleKernel<ULONG>::get_feature_type() { return F_ULONG; }

template<> inline EFeatureType CSimpleKernel<INT>::get_feature_type() { return F_INT; }

template<> inline EFeatureType CSimpleKernel<WORD>::get_feature_type() { return F_WORD; }

template<> inline EFeatureType CSimpleKernel<SHORT>::get_feature_type() { return F_SHORT; }

template<> inline EFeatureType CSimpleKernel<BYTE>::get_feature_type() { return F_BYTE; }

template<> inline EFeatureType CSimpleKernel<CHAR>::get_feature_type() { return F_CHAR; }


#endif
