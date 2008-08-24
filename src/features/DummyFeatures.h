/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008 Soeren Sonnenburg
 * Copyright (C) 2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _DUMMYFEATURES__H__
#define _DUMMYFEATURES__H__

#include "lib/common.h"
#include "features/Features.h"

class CDummyFeatures : public CFeatures
{
	public:
		CDummyFeatures(INT num) : CFeatures(0), num_vectors(num)
		{
		}

		CDummyFeatures(const CDummyFeatures &orig) : CFeatures(0),
			num_vectors(orig.num_vectors)
		{
		}

		virtual ~CDummyFeatures()
		{
		}

		virtual int get_num_vectors()
		{
			return num_vectors;
		}

		virtual INT get_size()
		{
			return 1;
		}

		virtual CFeatures* duplicate() const
		{
			return new CDummyFeatures(*this);
		}

		inline EFeatureType get_feature_type()
		{
			return F_ANY;
		}

		inline virtual EFeatureClass get_feature_class()
		{
			return C_ANY;
		}

	protected:
		INT num_vectors;
};
#endif
