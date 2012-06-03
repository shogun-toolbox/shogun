/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Sergey Lisitsyn
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#include <shogun/lib/IndicesTree.h>

using namespace shogun;

float64_t* CIndicesTree::get_ind() const
{
	SG_WARNING("Not implemented, uses one supernode\n");
	float64_t* ind = SG_MALLOC(float64_t, 3);
	ind[0] = -1;
	ind[1] = -1;
	ind[2] = 1.0;

	return ind;
}

