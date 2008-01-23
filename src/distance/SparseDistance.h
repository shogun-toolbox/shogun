/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007-2008 Soeren Sonnenburg
 * Copyright (C) 2007-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _SPARSEDISTANCE_H___
#define _SPARSEDISTANCE_H___

#include "distance/Distance.h"
#include "features/SparseFeatures.h"

template <class ST> class CSparseDistance : public CDistance
{
	public:
		CSparseDistance() : CDistance()
		{
		}

		/** initialize your kernel
		 * where l are feature vectors to occur on left hand side
		 * and r the feature vectors to occur on right hand side
		 *
		 */
		virtual bool init(CFeatures* l, CFeatures* r)
		{
			CDistance::init(l,r);

			ASSERT(l->get_feature_class() == C_SPARSE);
			ASSERT(r->get_feature_class() == C_SPARSE);
			ASSERT(l->get_feature_type()==this->get_feature_type());
			ASSERT(r->get_feature_type()==this->get_feature_type());

			if (((CSparseFeatures<ST>*) lhs)->get_num_features() != ((CSparseFeatures<ST>*) rhs)->get_num_features() )
			{
				SG_ERROR( "train or test features #dimension mismatch (l:%d vs. r:%d)\n",
						((CSparseFeatures<ST>*) lhs)->get_num_features(),((CSparseFeatures<ST>*)rhs)->get_num_features());
			}
			return true;
		}

		/** return feature class the kernel can deal with
		  */
		inline virtual EFeatureClass get_feature_class() { return C_SPARSE; }
		inline virtual EFeatureType get_feature_type();
};

template<> inline EFeatureType CSparseDistance<DREAL>::get_feature_type() { return F_DREAL; }

template<> inline EFeatureType CSparseDistance<ULONG>::get_feature_type() { return F_ULONG; }

template<> inline EFeatureType CSparseDistance<INT>::get_feature_type() { return F_INT; }

template<> inline EFeatureType CSparseDistance<WORD>::get_feature_type() { return F_WORD; }

template<> inline EFeatureType CSparseDistance<SHORT>::get_feature_type() { return F_SHORT; }

template<> inline EFeatureType CSparseDistance<BYTE>::get_feature_type() { return F_BYTE; }

template<> inline EFeatureType CSparseDistance<CHAR>::get_feature_type() { return F_CHAR; }
#endif
