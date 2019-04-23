/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Chiyuan Zhang, Bjoern Esser, Yuyu Zhang
 */

#ifndef BALANCEDCONDITIONALPROBABILITYTREE_H__
#define BALANCEDCONDITIONALPROBABILITYTREE_H__

#include <shogun/lib/config.h>

#include <shogun/multiclass/tree/ConditionalProbabilityTree.h>

namespace shogun
{

/**
 * Balanced Conditional Probability Tree.
 *
 * The tree is constructed to trade-off the existing regressor's prediction
 * and the balance (depth) of the tree. The parameter alpha in [0,1]
 * control the trade-off.
 *
 * * when alpha = 1, best efforts are made to ensure the tree is balanced
 * * when alpha = 0, the balance of tree is complete ignored
 *
 * more balanced tree means better computational efficiency, but usually worse
 * performance. See the following paper for more details:
 *
 *   Alina Beygelzimer, John Langford, Yuri Lifshits, Gregory Sorkin, Alex
 *   Strehl. Conditional Probability Tree Estimation Analysis and Algorithms. UAI 2009.
 */
class BalancedConditionalProbabilityTree: public ConditionalProbabilityTree
{
public:
	/** constructor */
	BalancedConditionalProbabilityTree();

	/** destructor */
	virtual ~BalancedConditionalProbabilityTree() {}

	/** get name */
	virtual const char* get_name() const { return "BalancedConditionalProbabilityTree"; }

	/** set alpha */
	void set_alpha(float64_t alpha);

	/** get alpha */
	float64_t get_alpha() const { return m_alpha; }

protected:
	/** decide which subtree to go, when training the tree structure.
	 * @param node the node being decided
	 * @param ex the example being decided
	 * @return true if should go left, false otherwise
	 */
	virtual bool which_subtree(std::shared_ptr<bnode_t> node, SGVector<float32_t> ex);

private:
	/** depth of subtree
	 * @param node pointer to the subtree root
	 * @return the depth of the subtree
	 */
	int32_t tree_depth(std::shared_ptr<bnode_t> node);

	/** trade-off parameter for tree balance */
	float64_t m_alpha;
};

} /* shogun */

#endif /* end of include guard: BALANCEDCONDITIONALPROBABILITYTREE_H__ */

