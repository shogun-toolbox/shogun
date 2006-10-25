/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _REALDISTANCE_H__
#define _REALDISTANCE_H__

#include "distance/SimpleDistance.h"
#include "lib/common.h"

class CRealDistance : public CSimpleDistance<DREAL>
{
	public:
		CRealDistance() : CSimpleDistance<DREAL>()
		{
		}

		virtual bool init(CFeatures* l, CFeatures* r, bool do_init)
		{
			CSimpleDistance<DREAL>::init(l,r, do_init);

			ASSERT(l->get_feature_type()==F_DREAL);
			ASSERT(r->get_feature_type()==F_DREAL);

			return true;
		}

		/** return feature type the distance can deal with
		  */
		inline virtual EFeatureType get_feature_type() { return F_DREAL; }
};
#endif
