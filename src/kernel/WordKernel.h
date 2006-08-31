/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _WORDKERNEL_H___
#define _WORDKERNEL_H___

#include "kernel/SimpleKernel.h"
#include "lib/common.h"

class CWordKernel : public CSimpleKernel<WORD>
{
	public:
		CWordKernel(LONG cachesize) : CSimpleKernel<WORD>(cachesize)
		{
		}

		virtual bool init(CFeatures* l, CFeatures* r, bool do_init)
		{
			CSimpleKernel<WORD>::init(l,r, do_init);

			ASSERT(l->get_feature_type()==F_WORD);
			ASSERT(r->get_feature_type()==F_WORD);

			return true;
		}

		/** return feature type the kernel can deal with
		  */
		inline virtual EFeatureType get_feature_type() { return F_WORD; }
};
#endif
