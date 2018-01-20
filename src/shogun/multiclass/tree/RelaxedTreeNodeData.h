/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Chiyuan Zhang, Soeren Sonnenburg, Yuyu Zhang, Björn Esser, 
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

	/** print data */
	static void print_data(const RelaxedTreeNodeData &data)
	{
		SG_SPRINT("left=(")
		for (int32_t i=0; i < data.mu.vlen; ++i)
			if (data.mu[i] == -1 || data.mu[i] == 0)
				SG_SPRINT("%4d", i)
		SG_SPRINT("), right=(")
		for (int32_t i=0; i < data.mu.vlen; ++i)
			if (data.mu[i] == 1 || data.mu[i] == 0)
				SG_SPRINT("%4d", i)
		SG_SPRINT(")\n")
	}
};

} /* shogun */

#endif /* end of include guard: RELAXEDTREENODEDATA_H__ */

