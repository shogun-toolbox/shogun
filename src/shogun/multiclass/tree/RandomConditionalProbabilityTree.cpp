/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#include <shogun/mathematics/Math.h>
#include <shogun/multiclass/tree/RandomConditionalProbabilityTree.h>

using namespace shogun;

bool CRandomConditionalProbabilityTree::which_subtree(bnode_t *node, SGVector<float32_t> ex)
{
	if (CMath::random(0.0, 1.0) > 0.5)
		return true;
	return false;
}
