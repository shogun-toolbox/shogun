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

#include <shogun/lib/config.h>

namespace shogun
{
/** @brief struct to store data of node of
 * conditional probability tree
 */
struct ConditionalProbabilityTreeNodeData
{
	/** labels */
	int32_t label;
	/** prob of right subtree used in prediction */
	float64_t p_right;

	/** constructor */
	ConditionalProbabilityTreeNodeData(): label(-1), p_right(0)
	{
	}

	/** print data */
	static void print_data(const ConditionalProbabilityTreeNodeData &data)
	{
		SG_SPRINT("label=%d\n", data.label)
	}
};


} /* shogun */

#endif /* end of include guard: CONDITIONALPROBABILITYTREENODEDATA_H__ */

