/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Chiyuan Zhang
 */

#include <shogun/mathematics/Math.h>
#include <shogun/multiclass/tree/RandomConditionalProbabilityTree.h>

using namespace shogun;

bool CRandomConditionalProbabilityTree::which_subtree(bnode_t *node, SGVector<float32_t> ex)
{
	if (m_rng->random(0.0, 1.0) > 0.5)
		return true;
	return false;
}
