/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Chiyuan Zhang, Soeren Sonnenburg, Yuyu Zhang, Bjoern Esser,
 *          Sergey Lisitsyn
 */

#ifndef RELAXEDTREENODEDATA_H__
#define RELAXEDTREENODEDATA_H__

#include <shogun/lib/config.h>

#include <shogun/lib/SGVector.h>

namespace shogun
{

/** Data for the tree nodes in a RelaxedTree */
struct RelaxedTreeNodeData
{
	/** mu */
	SGVector<int32_t> mu;
};

template<class T>
void register_params(RelaxedTreeNodeData& n, T* o)
{
	o->watch_param("mu", &n.mu, AnyParameterProperties("mu"));
}


} /* shogun */

#endif /* end of include guard: RELAXEDTREENODEDATA_H__ */

