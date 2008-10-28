/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _REALDISTANCE_H__
#define _REALDISTANCE_H__

#include "distance/SimpleDistance.h"
#include "lib/common.h"

/** class RealDistance */
class CRealDistance : public CSimpleDistance<float64_t>
{
	public:
		/** default constructor */
		CRealDistance() : CSimpleDistance<float64_t>() {}

		/** init distance
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if init was successful
		 */
		virtual bool init(CFeatures* l, CFeatures* r)
		{
			CSimpleDistance<float64_t>::init(l,r);

			ASSERT(l->get_feature_type()==F_DREAL);
			ASSERT(r->get_feature_type()==F_DREAL);

			return true;
		}

		/** get feature type the distance can deal with
		 *
		 * @return feature type DREAL
		 */
		inline virtual EFeatureType get_feature_type() { return F_DREAL; }
};
#endif
