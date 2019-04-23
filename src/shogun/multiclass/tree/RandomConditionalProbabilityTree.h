/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Chiyuan Zhang, Yuyu Zhang, Bjoern Esser
 */

#ifndef RANDOMCONDITIONALPROBABILITYTREE_H__
#define RANDOMCONDITIONALPROBABILITYTREE_H__

#include <shogun/lib/config.h>

#include <shogun/multiclass/tree/ConditionalProbabilityTree.h>
#include <shogun/mathematics/RandomMixin.h>
#include <shogun/mathematics/UniformRealDistribution.h>

namespace shogun
{

/** Conditional Probability Tree, decide subtree by a random strategy.
 */
class RandomConditionalProbabilityTree: public RandomMixin<ConditionalProbabilityTree>
{
public:
    /** constructor */
	RandomConditionalProbabilityTree() : m_uniform_real_dist(0.0, 1.0) {}

    /** destructor */
	virtual ~RandomConditionalProbabilityTree() {}

    /** get name */
    virtual const char* get_name() const { return "RandomConditionalProbabilityTree"; }

protected:
	/** decide which subtree to go, when training the tree structure.
	 * @param node the node being decided
	 * @param ex the example being decided
	 * @return true if should go left, false otherwise
	 */
	virtual bool which_subtree(std::shared_ptr<bnode_t> node, SGVector<float32_t> ex);

private:
	UniformRealDistribution<float64_t> m_uniform_real_dist;
};

} /* shogun */

#endif /* end of include guard: RANDOMCONDITIONALPROBABILITYTREE_H__ */

