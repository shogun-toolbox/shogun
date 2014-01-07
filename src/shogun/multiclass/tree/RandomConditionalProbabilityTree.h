/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#ifndef RANDOMCONDITIONALPROBABILITYTREE_H__
#define RANDOMCONDITIONALPROBABILITYTREE_H__

#include <multiclass/tree/ConditionalProbabilityTree.h>

namespace shogun
{

/** Conditional Probability Tree, decide subtree by a random strategy.
 */
class CRandomConditionalProbabilityTree: public CConditionalProbabilityTree
{
public:
    /** constructor */
	CRandomConditionalProbabilityTree() {}

    /** destructor */
	virtual ~CRandomConditionalProbabilityTree() {}

    /** get name */
    virtual const char* get_name() const { return "RandomConditionalProbabilityTree"; }

protected:
	/** decide which subtree to go, when training the tree structure.
	 * @param node the node being decided
	 * @param ex the example being decided
	 * @return true if should go left, false otherwise
	 */
	virtual bool which_subtree(node_t *node, SGVector<float32_t> ex);
};

} /* shogun */

#endif /* end of include guard: RANDOMCONDITIONALPROBABILITYTREE_H__ */

