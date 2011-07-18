/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "features/Features.h"
#include "features/AttributeFeatures.h"

using namespace shogun;

CAttributeFeatures::CAttributeFeatures()
: CFeatures(0)
{
}

CAttributeFeatures::~CAttributeFeatures()
{
	int32_t n=features.get_num_elements();
	for (int32_t i=0; i<n; i++)
		SG_UNREF_NO_NULL(features[i].attr_obj);
}
