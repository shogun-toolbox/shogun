/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#ifndef CONDITIONALPROBABILITYTREE_H__
#define CONDITIONALPROBABILITYTREE_H__

#include <map>

#include <shogun/multiclass/tree/TreeMachine.h>
#include <shogun/classifier/vw/VowpalWabbit.h>

namespace shogun
{

class CConditionalProbabilityTree: public CTreeMachine
{
public:
    /** constructor */
	CConditionalProbabilityTree(int32_t num_passes=2)
		:m_num_passes(num_passes)
	{
	}

    /** destructor */
	virtual ~CConditionalProbabilityTree() {}

    /** get name */
    virtual const char* get_name() const { return "ConditionalProbabilityTree"; }

	/** set number of passes */
	void set_num_passes(int32_t num_passes)
	{
		m_num_passes = num_passes;
	}

	/** get number of passes */
	int32_t get_num_passes() const
	{
		return m_num_passes;
	}

protected:
	/** train machine
	 *
	 * @param data training data 
	 *
	 * @return whether training was successful
	 */
	virtual bool train_machine(CFeatures* data);

	/** train on a single example (online learning)
	 * @param ex VwExample instance
	 */
	void train_example(VwExample *ex);

	/** train on a path from a node up to the root
	 * @param ex VwExample instance of the training example
	 * @param node the leaf node
	 */
	void train_path(VwExample *ex, CTreeMachineNode *node);

	/** train a single node 
	 * @param ex VwExample instance of the training example
	 * @param node the node
	 */
	void train_node(VwExample *ex, CTreeMachineNode *node);

	/** create a new VW machine for a node 
	 * @param ex the VwExample instance for training the new machine
	 */
	int32_t create_machine(VwExample *ex);

	/** decide which subtree to go, when training the tree structure.
	 * @param node the node being decided
	 * @param ex the example being decided
	 * @return true if should go left, false otherwise
	 */
	virtual bool which_subtree(CTreeMachineNode *node, VwExample *ex)=0;

	int32_t m_num_passes; ///< number of passes for online training
	std::map<int32_t, CTreeMachineNode*> m_leaves;
};

} /* shogun */ 

#endif /* end of include guard: CONDITIONALPROBABILITYTREE_H__ */

