/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006 Christian Gehl
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _STRINGDISTANCE_H___
#define _STRINGDISTANCE_H___

#include "distance/Distance.h"
#include "features/StringFeatures.h"

template <class ST> class CStringDistance : public CDistance
{
	public:
		CStringDistance() : CDistance()
		{
		}

		/** initialize your distance
		 * where l are feature vectors to occur on left hand side
		 * and r the feature vectors to occur on right hand side
		 *
		 * when training data is supplied as both l and r do_init should
		 * be true
		 */
		
		virtual bool init(CFeatures* l, CFeatures* r)
		{
			CDistance::init(l,r);

			ASSERT(l->get_feature_class() == C_STRING);
			ASSERT(r->get_feature_class() == C_STRING);
			ASSERT(l->get_feature_type()==this->get_feature_type());
			ASSERT(r->get_feature_type()==this->get_feature_type());
			return true;
		}

		/** return feature class the distance can deal with
		  */
		inline virtual EFeatureClass get_feature_class() { return C_STRING; }
		/** return feature type the distance can deal with
		  */
		virtual EFeatureType get_feature_type();
};

template<> inline EFeatureType CStringDistance<DREAL>::get_feature_type() { return F_DREAL; }

template<> inline EFeatureType CStringDistance<ULONG>::get_feature_type() { return F_ULONG; }

template<> inline EFeatureType CStringDistance<INT>::get_feature_type() { return F_INT; }

template<> inline EFeatureType CStringDistance<WORD>::get_feature_type() { return F_WORD; }

template<> inline EFeatureType CStringDistance<SHORT>::get_feature_type() { return F_SHORT; }

template<> inline EFeatureType CStringDistance<BYTE>::get_feature_type() { return F_BYTE; }

template<> inline EFeatureType CStringDistance<CHAR>::get_feature_type() { return F_CHAR; }

#endif

