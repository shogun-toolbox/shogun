/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#include <shogun/multiclass/tree/RandomConditionalProbabilityTree.h>

using namespace shogun;


bool CRandomConditionalProbabilityTree::train_machine(CFeatures* data)
{
	if (!data)
		SG_ERROR("No data provided\n");
	if (data->get_feature_class() != C_STREAMING_VW)
		SG_ERROR("Expected StreamingVwFeatures\n");

	
}

