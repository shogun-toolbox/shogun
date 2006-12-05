/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006 Christian Gehl
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _STRINGDISTANCE_H___
#define _STRINGDISTANCE_H___

#include "distance/Distance.h"
#include "features/StringFeatures.h"

template <class ST> class CStringDistance : public CDistance
{
	public:
		CStringDistance(LONG cachesize) : CDistance(cachesize)
		{
		}

		/** initialize your distance
		 * where l are feature vectors to occur on left hand side
		 * and r the feature vectors to occur on right hand side
		 *
		 * when training data is supplied as both l and r do_init should
		 * be true
		 */
		
		virtual bool init(CFeatures* l, CFeatures* r, bool do_init)
		{
			CDistance::init(l,r,do_init);

			ASSERT(l->get_feature_class() == C_STRING);
			ASSERT(r->get_feature_class() == C_STRING);

			return true;
		}

		/** return feature class the distance can deal with
		  */
		inline virtual EFeatureClass get_feature_class() { return C_STRING; }
};
#endif

