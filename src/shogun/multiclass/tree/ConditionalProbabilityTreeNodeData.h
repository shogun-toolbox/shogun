/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Soeren Sonnenburg, Yuyu Zhang, Björn Esser, 
 *          Chiyuan Zhang
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

