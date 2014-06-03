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


#ifndef _C45CLASSIFIERTREE_H__
#define _C45CLASSIFIERTREE_H__

#include <shogun/lib/config.h>

#include <shogun/multiclass/tree/TreeMachine.h>
#include <shogun/multiclass/tree/C45TreeNodeData.h>
#include <shogun/features/DenseFeatures.h>

namespace shogun
{

/** @brief Class C45ClassifierTree implements the C4.5 algorithm for decision tree
 * learning. The algorithm steps are briefy explained below : \n
 *
 * function C4.5 (R: a set of non-categorical attributes, C: the categorical attribute, S: a training set):\n
 *    returns a decision tree; \n
 *
 * begin \n
 *    If S consists of records all with the same value for the categorical attribute,\n
 *      return a single node with that value; \n
 *
 *    If R is empty,\n
 *      return a single node with as value the most frequent\n 
 *      of the values of the categorical attribute in C;\n 
 *      [note that then there will be errors, that is, records that will be improperly classified]; \n
 *
 *    For each non-categorical attribute NC in R : \n
 *      If NC is continuous then first convert it to nominal attribute by separating into 2 classes about a threshold \n
 *      Find Gain of all attributes \n
 *
 *    Let D be the attribute with largest Gain(D,S) among attributes in R;\n
 *    Let \f${d_j| j=1,2, .., m}\f$ be the values of attribute D;\n
 *    Let \f${S_j| j=1,2, .., m}\f$ be the subsets of S consisting respectively of records with value \f$d_j\f$ 
 * for attribute D; \n
 *
 *    Return a tree with root labeled D and arcs labeled \f$d_1, d_2, .., d_m\f$ going respectively to the trees \n
 *
 *    C4.5(R-{D}, C, \f$S_1\f$), .., C4.5(R-{D}, C, \f$S_m\f$);\n
 * end C4.5;
 *
 * cf. http://tesis-algoritmo-c45.googlecode.com/files/C45.ppt
 */
class CC45ClassifierTree : public CTreeMachine<C45TreeNodeData>
{
public:
	/** constructor */
	CC45ClassifierTree();

	/** destructor */
	virtual ~CC45ClassifierTree();

	/** get name
	 * @return class name C45ClassifierTree
	 */
	virtual const char* get_name() const { return "C45ClassifierTree"; }

	/** classify data using C4.5 Tree
	 * @param data data to be classified
	 * @return MulticlassLabels corresponding to labels of various test vectors 
	 */
	virtual CMulticlassLabels* apply_multiclass(CFeatures* data=NULL);

	/** prune decision tree - uses reduced error pruning algorithm
	 * 
	 * cf. http://en.wikipedia.org/wiki/Pruning_%28decision_trees%29#Reduced_error_pruning
	 *
	 * At each node, starting from leaf nodes up to the root node, this
	 * algorithm checks if removing the subtree gives better results (or
	 * somewhat comparable results). If so, it replaces the subtree with
	 * a leaf node. The algorithm implemented is recursive which starts with
	 * the root node. At each node, it prunes its children first and then itself.
	 * As the algorithm goes down each level during recursion, it creates the new
	 * set of features by pushing subset into subset stack. While retracting, it pops
	 * these subsets to access previous state of feature matrix (see add_subset() and 
	 * remove_subset() in Shogun documentation). 
	 *
	 * @param validation_data feature vectors from validation dataset
	 * @param validation_labels multiclass labels from validation dataset
	 * @param epsilon prune subtree even if there is epsilon loss in accuracy
	 */
	void prune_tree(CDenseFeatures<float64_t>* validation_data, CMulticlassLabels* validation_labels, float64_t epsilon=0.f);

	/** certainty of classification done by apply_multiclass.
	 * For each data point reaching a leaf node, it computes the ratio of weight of training data 
	 * with same predicted label that reached that leaf node over the total weight of all training data points
	 * that reached the same leaf node
	 *
	 * @return Vector of certainty values associated with data classified in apply_multiclass 
	 */
	SGVector<float64_t> get_certainty_vector() const;

	/** set weights of data points
	 * @param w vector of weights 
	 */
	void set_weights(SGVector<float64_t> w);

	/** get weights of data points
	 * @return vector of weights 
	 */
	SGVector<float64_t> get_weights() const;

	/** clear weights of data points */
	void clear_weights();

	/** set feature types of various features
	 * @param ft bool vector true for nominal feature false for continuous feature type 
	 */
	void set_feature_types(SGVector<bool> ft);

	/** set feature types of various features
	 * @return bool vector - true for nominal feature false for continuous feature type 
	 */
	SGVector<bool> get_feature_types() const;

	/** clear feature types of various features */
	void clear_feature_types();

protected:
	
	/** train machine - build C4.5 Tree from training data
	 * @param data training data
	 */
	virtual bool train_machine(CFeatures* data=NULL);

private:

	/** C45train - recursive C45 training method
	 *
	 * @param data training data
	 * @param weights vector of weights of data points
	 * @param class_labels labels of data points
	 * @param id_values vector of id numbers of features in data
	 * @param level tree depth
	 * @return pointer to the root of the C45 subtree
	 */
	node_t* C45train(CFeatures* data, SGVector<float64_t> weights, CMulticlassLabels* class_labels, 
							SGVector<int32_t> id_values, int level = 0);

	/** recursive tree pruning method - called within prune_tree method
	 *
	 * @param feats feature set to use for pruning
	 * @param gnd_truth ground truth labels
	 * @param current root of current subtree
	 * @param epsilon prune subtree even if there is epsilon loss in accuracy
	 */
	void prune_tree_from_current_node(CDenseFeatures<float64_t>* feats, CMulticlassLabels* gnd_truth, node_t* current, float64_t epsilon);
	
	/** informational gain attribute for selecting best feature at each node of C4.5 Tree
	 *
	 * @param attr_no index to the chosen feature in data matrix supplied
	 * @param data data matrix
	 * @param weights weights of data points in data matrix
	 * @param class_labels classes to which corresponding data vectors belong
	 * @return informational gain of the chosen feature
	 */	
	float64_t informational_gain_attribute(int32_t attr_no, CFeatures* data, SGVector<float64_t> weights,
									 CMulticlassLabels* class_labels);	
	
	/** computes entropy (aka randomness) in data
	 *
	 * @param labels labels of parameters chosen
	 * @param weights weights associated with each of the labels 
	 * @return entropy
	 */		
	float64_t entropy(CMulticlassLabels* labels, SGVector<float64_t> weights);

	/** uses current subtree to classify data
	 *
	 * @param feats data to be classified
	 * @param current root of current subtree
	 * @param set_certainty whether to calculate certainty values or not
	 * @return classification labels of input data
	 */
	CMulticlassLabels* apply_multiclass_from_current_node(CDenseFeatures<float64_t>* feats, node_t* current, bool set_certainty=false);

	/** initializes members of class */
	void init();

public:
	/** denotes that a feature in a vector is missing MISSING = NOT_A_NUMBER */
	static const float64_t MISSING;

private:

	/** vector depicting whether various feature dimensions are nominal or not **/
	SGVector<bool> m_nominal;

	/** weights of samples in training set **/
	SGVector<float64_t> m_weights;

	/** percentage of certainty of labels predicted by decision tree 
	 * ie. weight of elements belonging to predicted class in a node/ total weight in a node
	 */
	SGVector<float64_t> m_certainty;

	/** flag storing whether the type of various feature dimensions are specified using is_nominal_feature **/
	bool m_types_set;

	/** flag storing whether weights of samples are specified using weights vector **/
	bool m_weights_set;
	
};
} /* namespace shogun */

#endif /* _C45CLASSIFIERTREE_H__ */
