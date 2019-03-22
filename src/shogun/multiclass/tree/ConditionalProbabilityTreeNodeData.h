/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Soeren Sonnenburg, Yuyu Zhang, Bjoern Esser,
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
};

template<class T>
constexpr void register_params(ConditionalProbabilityTreeNodeData& n, T* o)
{
	o->watch_param("label", &n.label, AnyParameterProperties("label"));
	o->watch_param("p_right", &n.p_right, AnyParameterProperties("prob of right subtree used in prediction"));
}


} /* shogun */

#endif /* end of include guard: CONDITIONALPROBABILITYTREENODEDATA_H__ */

