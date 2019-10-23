/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Chiyuan Zhang, Sergey Lisitsyn, Yuyu Zhang, Viktor Gal,
 *          Fernando Iglesias, Bjoern Esser
 */

#ifndef CONDITIONALPROBABILITYTREE_H__
#define CONDITIONALPROBABILITYTREE_H__

#include <map>

#include <shogun/lib/config.h>

#include <shogun/features/streaming/StreamingDenseFeatures.h>
#include <shogun/multiclass/tree/TreeMachine.h>
#include <shogun/multiclass/tree/ConditionalProbabilityTreeNodeData.h>

namespace shogun
{

/**
 * Conditional Probability Tree.
 *
 * See reference:
 *
 *   Alina Beygelzimer, John Langford, Yuri Lifshits, Gregory Sorkin, Alex
 *   Strehl. Conditional Probability Tree Estimation Analysis and Algorithms. UAI 2009.
 */
class ConditionalProbabilityTree: public TreeMachine<ConditionalProbabilityTreeNodeData>
{
public:
    /** constructor */
	ConditionalProbabilityTree(int32_t num_passes=1)
		:m_num_passes(num_passes), m_feats(NULL)
	{
	}

    /** destructor */
	virtual ~ConditionalProbabilityTree() {  }

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

	/** set features
	 * @param feats features
	 */
	void set_features(std::shared_ptr<StreamingDenseFeatures<float32_t> >feats)
	{


		m_feats = feats;
	}

	/** apply machine to data in means of multiclass classification problem */
	virtual std::shared_ptr<MulticlassLabels> apply_multiclass(std::shared_ptr<Features> data=NULL);

	/** apply machine one single example.
	 * @param ex a vector to be applied
	 */
	virtual int32_t apply_multiclass_example(SGVector<float32_t> ex);

	/** the labels will be embedded in the streaming features */
	virtual bool train_require_labels() const { return false; }

protected:
	/** train machine
	 *
	 * @param data training data
	 *
	 * @return whether training was successful
	 */
	virtual bool train_machine(std::shared_ptr<Features> data);

	/** train on a single example (online learning)
	 * @param ex the example being trained
	 * @param label the label of this training example
	 */
	void train_example(const std::shared_ptr<StreamingDenseFeatures<float32_t>>& ex, int32_t label);

	/** train on a path from a node up to the root
	 * @param ex the instance of the training example
	 * @param node the leaf node
	 */
	void train_path(const std::shared_ptr<StreamingDenseFeatures<float32_t>>& ex, std::shared_ptr<bnode_t> node);

	/** train a single node
	 * @param ex the example being trained
	 * @param label label
	 * @param node the node
	 */
	void train_node(const std::shared_ptr<StreamingDenseFeatures<float32_t>>& ex, float64_t label, const std::shared_ptr<bnode_t>& node);

	/** predict a single node
	 * @param ex the example being predicted
	 * @param node the node
	 */
	float64_t predict_node(SGVector<float32_t> ex, const std::shared_ptr<bnode_t>& node);

	/** create a new OnlineLinear machine for a node
	 * @param ex the Example instance for training the new machine
	 */
	int32_t create_machine(const std::shared_ptr<StreamingDenseFeatures<float32_t>>& ex);

	/** decide which subtree to go, when training the tree structure.
	 * @param node the node being decided
	 * @param ex the example being decided
	 * @return true if should go left, false otherwise
	 */
	virtual bool which_subtree(std::shared_ptr<bnode_t> node, SGVector<float32_t> ex)=0;

	/** compute conditional probabilities for ex along the whole tree for predicting */
	void compute_conditional_probabilities(const SGVector<float32_t>& ex);

	/** accumulate along the path to the root the conditional probability for a
	 * particular leaf node.
	 */
	float64_t accumulate_conditional_probability(std::shared_ptr<bnode_t> leaf);

	int32_t m_num_passes; ///< number of passes for online training
	std::map<int32_t, std::shared_ptr<bnode_t>> m_leaves; ///< class => leaf mapping
	std::shared_ptr<StreamingDenseFeatures<float32_t> >m_feats; ///< online features
};

} /* shogun */

#endif /* end of include guard: CONDITIONALPROBABILITYTREE_H__ */

