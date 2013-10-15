/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#ifndef BALANCEDCONDITIONALPROBABILITYTREE_H__
#define BALANCEDCONDITIONALPROBABILITYTREE_H__

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
class CBalancedConditionalProbabilityTree: public CConditionalProbabilityTree
{
public:
	/** constructor */
	CBalancedConditionalProbabilityTree();

	/** destructor */
	virtual ~CBalancedConditionalProbabilityTree() {}

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
	virtual bool which_subtree(node_t *node, SGVector<float32_t> ex);

private:
	int32_t tree_depth(node_t *node);

	float64_t m_alpha; ///< trade-off parameter for tree balance
};

} /* shogun */

#endif /* end of include guard: BALANCEDCONDITIONALPROBABILITYTREE_H__ */

