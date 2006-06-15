/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Written (W) 1999-2006 Gunnar Raetsch
 * Written (W) 1999-2006 Fabio De Bona
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _SIMPLEKERNEL_H___
#define _SIMPLEKERNEL_H___

#include "kernel/Kernel.h"
#include "features/SimpleFeatures.h"
#include "lib/io.h"

template <class ST> class CSimpleKernel : public CKernel
{
	public:
		CSimpleKernel(LONG cachesize) : CKernel(cachesize)
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

			if ( (l->get_feature_class() != C_SIMPLE) ||
					(r->get_feature_class() != C_SIMPLE) ||
					((CSimpleFeatures<ST>*) l)->get_num_features() != ((CSimpleFeatures<ST>*) r)->get_num_features() )
			{
				CIO::message(M_ERROR, "train or test features not of type SIMPLE, or #features mismatch (l:%d vs. r:%d)\n",
						((CSimpleFeatures<ST>*) l)->get_num_features(),((CSimpleFeatures<ST>*) l)->get_num_features());
			}
			return true;
		}

		/** return feature class the kernel can deal with
		  */
		inline virtual EFeatureClass get_feature_class() { return C_SIMPLE; }
};
#endif
