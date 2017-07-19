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
	std::uniform_real_distribution<float64_t> dist(0.0, 1.0);
	if (dist(m_rng) > 0.5)
		return true;
	return false;
}
