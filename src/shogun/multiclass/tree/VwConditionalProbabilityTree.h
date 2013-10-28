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

/** Tree Data for Vowpal Wabbit Conditional Probability Tree */
struct VwConditionalProbabilityTreeNodeData
{
	/** actual label */
	int32_t label;
	/** probability of right subtree used for prediction */
	float64_t p_right;

	/** ctor */
	VwConditionalProbabilityTreeNodeData():label(-1), p_right(0) {}
};

/** define shortcut for the node type */
typedef CTreeMachineNode<VwConditionalProbabilityTreeNodeData> node_t;

/** Conditional Probability Tree using Vowpal Wabbit for base machine. */
class CVwConditionalProbabilityTree: public CTreeMachine<VwConditionalProbabilityTreeNodeData>
{
public:

    /** constructor */
	CVwConditionalProbabilityTree(int32_t num_passes=1)
		:m_num_passes(num_passes), m_feats(NULL)
	{
	}

    /** destructor */
	virtual ~CVwConditionalProbabilityTree() {}

    /** get name */
    virtual const char* get_name() const { return "VwConditionalProbabilityTree"; }

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

	/** set features
	 * @param feats features
	 */
	void set_features(CStreamingVwFeatures *feats)
	{
		SG_REF(feats);
		SG_UNREF(m_feats);
		m_feats = feats;
	}

	/** apply machine to data in means of multiclass classification problem */
	virtual CMulticlassLabels* apply_multiclass(CFeatures* data=NULL);

	/** apply machine one single example */
	virtual int32_t apply_multiclass_example(VwExample* ex);
protected:
	/** the labels will be embedded in the streaming features */
	virtual bool train_require_labels() const { return false; }

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
	void train_path(VwExample *ex, node_t *node);

	/** train a single node
	 * @param ex VwExample instance of the training example
	 * @param node the node
	 * @return the predicted value for the example
	 */
	float64_t train_node(VwExample *ex, node_t *node);

	/** create a new VW machine for a node
	 * @param ex the VwExample instance for training the new machine
	 */
	int32_t create_machine(VwExample *ex);

	/** decide which subtree to go, when training the tree structure.
	 * @param node the node being decided
	 * @param ex the example being decided
	 * @return true if should go left, false otherwise
	 */
	virtual bool which_subtree(node_t *node, VwExample *ex)=0;

	/** compute conditional probabilities for ex along the whole tree for predicting */
	void compute_conditional_probabilities(VwExample *ex);

	/** accumulate along the path to the root the conditional probability for a
	 * particular leaf node.
	 */
	float64_t accumulate_conditional_probability(node_t *leaf);

	int32_t m_num_passes; ///< number of passes for online training
	std::map<int32_t, node_t*> m_leaves; ///< class => leaf mapping
	CStreamingVwFeatures *m_feats; ///< online features
};

} /* shogun */

#endif /* end of include guard: CONDITIONALPROBABILITYTREE_H__ */

