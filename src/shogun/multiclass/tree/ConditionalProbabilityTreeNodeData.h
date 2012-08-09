/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#ifndef CONDITIONALPROBABILITYTREENODEDATA_H__
#define CONDITIONALPROBABILITYTREENODEDATA_H__

namespace shogun
{

struct ConditionalProbabilityTreeNodeData
{
	int32_t label;
	float64_t p_right; // prob of right subtree, used in prediction

	ConditionalProbabilityTreeNodeData():label(-1), p_right(0) {}

	static void print(const ConditionalProbabilityTreeNodeData &data)
	{
		SG_SPRINT("label=%d\n", data.label);
	}
};


} /* shogun */ 

#endif /* end of include guard: CONDITIONALPROBABILITYTREENODEDATA_H__ */

