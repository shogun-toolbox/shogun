/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Parijat Mazumdar
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 */


#ifndef _ID3CLASSIFIERTREE_H__
#define _ID3CLASSIFIERTREE_H__

#include <shogun/lib/config.h>

#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/lib/SGVector.h>
#include <shogun/multiclass/tree/FeatureImportanceTree.h>
#include <shogun/multiclass/tree/ID3TreeNodeData.h>
#include <shogun/multiclass/tree/TreeMachine.h>

namespace shogun
{

/** @brief class ID3ClassifierTree, implements classifier tree for discrete feature
 * values using the ID3 algorithm. The training algorithm implemented is as follows :
 *
 * function ID3 (R: a set of features, C: the Labels, S: a training set)\n
 * returns a decision tree;
 *
 * begin \n
 *	If S consists of records all with the same value for the Label,\n
 *	   return a single node with that value;
 *
 *	If R is empty,\n
 * 	   return a single node with as value the most frequent
 * of the values of the Label that are found in records of S;\n
 * [note that then there will be errors, that is, records that will be improperly classified];
 *
 *	Let D be the attribute with largest Gain(D,S)
 * among attributes in R;
 *
 *	Let \f${d_j| j=1,2, .., m}\f$ be the values of attribute D;\n
 *	Let \f${S_j| j=1,2, .., m}\f$ be the subsets of S consisting
 * respectively of records with value dj for attribute D;
 *
 *	Return a tree with root labeled D and arcs labeled
 * \f$d_1, d_2, .., d_m\f$ going respectively to the trees \n
 *	ID3(R-{D}, C, \f$S_1\f$), .., ID3(R-{D}, C, \f$S_m\f$);
 *
 * end ID3;
 *
 */
class ID3ClassifierTree : public FeatureImportanceTree<id3TreeNodeData>
{
public:
	/** constructor */
	ID3ClassifierTree();

	/** destructor */
	~ID3ClassifierTree() override;

	/** get name
	 * @return class name ID3ClassifierTree
	 */
	const char* get_name() const override { return "ID3ClassifierTree"; }

	/** classify data using ID3 Tree
	 * @param data data to be classified
	 */
	std::shared_ptr<MulticlassLabels> apply_multiclass(std::shared_ptr<Features> data=NULL) override;

	/** prune id3 decision tree - uses reduced error pruning algorithm
	 *
	 * cf.
	 * http://en.wikipedia.org/wiki/Pruning_%28decision_trees%29#Reduced_error_pruning
	 *
	 * At each node, starting from leaf nodes up to the root node, this
	 * algorithm checks if removing the subtree gives better results (or
	 * somewhat comparable results). If so, it replaces the subtree with
	 * a leaf node. The algorithm implemented is recursive which starts with
	 * the root node. At each node, it prunes its children first and then
	 * itself. As the algorithm goes down each level during recursion, it
	 * creates a view of features which is a subset of original features.
	 *
	 * @param validation_data feature vectors from validation dataset
	 * @param validation_labels multiclass labels from validation dataset
	 * @param epsilon prune subtree even if there is epsilon loss in accuracy
	 *
	 * @return true if pruning successful
	 */
	bool prune_tree(std::shared_ptr<DenseFeatures<float64_t>> validation_data, std::shared_ptr<MulticlassLabels> validation_labels, float64_t epsilon=0.f);

	SGVector<float64_t> get_feature_importances() const;

protected:

	/** train machine - build ID3 Tree from training data
	 * @param data training data
	 */
	bool train_machine(std::shared_ptr<Features> data=NULL) override;

private:

	/** id3train - recursive id3 training method
	 *
	 * @param data training data
	 * @param class_labels labels associated with data
	 * @param values feature indices left in current node
	 * @param level current depth of tree
	 * @return pointer to the root of the ID3 tree
	 */
	std::shared_ptr<node_t> id3train(const std::shared_ptr<Features>& data, const std::shared_ptr<MulticlassLabels>& class_labels, SGVector<int32_t> values, int level = 0);

	/** informational gain attribute for selecting best feature at each node of ID3 Tree
	 *
	 * @param attr_no index to the chosen feature in data matrix supplied
	 * @param data data matrix
	 * @param class_labels classes to which corresponding data vectors belong
	 * @return informational gain of the chosen feature
	 */
	float64_t informational_gain_attribute(
		int32_t attr_no, const std::shared_ptr<Features>& data,
		const std::shared_ptr<MulticlassLabels>& class_labels,
		float64_t& impurity);

	/** computes entropy (aka randomness) in data
	 *
	 * @param labels labels of parameters chosen
	 * @return entropy
	 */
	float64_t entropy(const std::shared_ptr<MulticlassLabels>& labels);

	/** recursive tree pruning method - called within prune_tree method
	 *
	 * @param feats feature set to use for pruning
	 * @param gnd_truth ground truth labels
	 * @param current root of current subtree
	 * @param epsilon prune subtree even if there is epsilon loss in accuracy
	 */
	void prune_tree_machine(const std::shared_ptr<DenseFeatures<float64_t>>& feats, const std::shared_ptr<MulticlassLabels>& gnd_truth, const std::shared_ptr<node_t>& current, float64_t epsilon);

	/** uses current subtree to classify data
	 *
	 * @param feats data to be classified
	 * @param current root of current subtree
	 * @return classification labels of input data
	 */
	std::shared_ptr<MulticlassLabels> apply_multiclass_from_current_node(const std::shared_ptr<DenseFeatures<float64_t>>& feats, const std::shared_ptr<node_t>& current);
};
} /* namespace shogun */

#endif /* _ID3CLASSIFIERTREE_H__ */
