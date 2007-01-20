/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006 Christian Gehl
 * Copyright (C) 2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _SIMPLEDISTANCE_H___
#define _SIMPLEDISTANCE_H___

#include "distance/Distance.h"
#include "features/SimpleFeatures.h"
#include "lib/io.h"

template <class ST> class CSimpleDistance : public CDistance
{
	public:
		CSimpleDistance() : CDistance()
		{
		}

		/** initialize your distance
		 * where l are feature vectors to occur on left hand side
		 * and r the feature vectors to occur on right hand side
		 *
		 */
		virtual bool init(CFeatures* l, CFeatures* r, bool do_init)
		{
			CDistance::init(l,r,do_init);

			if ( (l->get_feature_class() != C_SIMPLE) ||
					(r->get_feature_class() != C_SIMPLE) ||
					((CSimpleFeatures<ST>*) l)->get_num_features() != ((CSimpleFeatures<ST>*) r)->get_num_features() )
			{
				SG_ERROR( "train or test features not of type SIMPLE, or #features mismatch (l:%d vs. r:%d)\n",
						((CSimpleFeatures<ST>*) l)->get_num_features(),((CSimpleFeatures<ST>*) l)->get_num_features());
			}
			return true;
		}

		/** return feature class the distance can deal with
		  */
		inline virtual EFeatureClass get_feature_class() { return C_SIMPLE; }
};
#endif
