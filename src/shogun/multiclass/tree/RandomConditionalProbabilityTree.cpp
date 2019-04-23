/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Chiyuan Zhang
 */

#include <shogun/mathematics/Math.h>
#include <shogun/multiclass/tree/RandomConditionalProbabilityTree.h>
#include <shogun/mathematics/RandomNamespace.h>

using namespace shogun;

bool RandomConditionalProbabilityTree::which_subtree(std::shared_ptr<bnode_t> node, SGVector<float32_t> ex)
{
	if (m_uniform_real_dist(m_prng) > 0.5)
		return true;
	return false;
}
