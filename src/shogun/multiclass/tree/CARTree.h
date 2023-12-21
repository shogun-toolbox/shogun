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


#ifndef _CARTREE_H__
#define _CARTREE_H__

#include <memory>
#include <shogun/lib/config.h>

#include <shogun/features/DenseFeatures.h>
#include <shogun/features/DenseSubSamplesFeatures.h>
#include <shogun/mathematics/RandomMixin.h>
#include <shogun/multiclass/tree/CARTreeNodeData.h>
#include <shogun/multiclass/tree/FeatureImportanceTree.h>
#include <shogun/multiclass/tree/TreeMachine.h>

#include <vector>

namespace shogun
{

/** @brief This class implements the Classification And Regression Trees algorithm by Breiman et al for decision tree learning.
 * A CART tree is a binary decision tree that is constructed by splitting a node into two child nodes repeatedly, beginning with
 * the root node that contains the whole dataset. \n \n
 * TREE GROWING PROCESS : \n
 * During the tree growing process, we recursively split a node into left child and right child so that the resulting nodes are "purest".
 * We do this until any of the stopping criteria is met. To find the best split, we scan through all possible splits in all predictive
 * attributes. The best split is one that maximises some splitting criterion. For classification tasks, ie. when the dependent attribute
 * is categorical, the Gini index is used. For regression tasks, ie. when the dependent variable is continuous, least squares deviation is
 * used. The algorithm uses two stopping criteria : if node becomes completely "pure", ie. all its members have identical dependent
 * variable, or all of them have identical predictive attributes (independent variables). \n \n
 *
 * COST-COMPLEXITY PRUNING : \n
 * The maximal tree, \f$T_max\f$ grown during tree growing process is bound to overfit. Hence pruning becomes necessary. Cost-Complexity
 * pruning yields a list of subtrees of varying depths using the complexity normalized resubstitution error, \f$R_\alpha(T)\f$. The
 * resubstitution error R(T) is a measure of how well a decision tree fits the training data. This measure favours larger trees over
 * smaller ones. However, complexity normalized resubstitution error, adds penalty for increased complexity and hence counters overfitting.\n
 * \f$R_\alpha(T)=R(T)+\alpha \times (numleaves)\f$ \n
 * The best subtree among the list of subtrees can be chosen using cross validation or using best-fit in the test dataset. \n
 * cf. https://onlinecourses.science.psu.edu/stat557/node/93 \n \n
 *
 * HANDLING MISSING VALUES : \n
 * While choosing the best split at a node, missing attribute values are left out. But data vectors with missing values of the best attribute
 * chosen are sent to left child or right child using a surrogate split. A surrogate split is one that imitates the best split as closely
 * as possible. While choosing a surrogate split, all splits alternative to the best split are scaned and the degree of closeness between the
 * two is measured using a metric called predictive measure of association, \f$\lambda_{i,j}\f$. \n
 * \f$\lambda_{i,j} = \frac{min(P_L,P_R)-(1-P_{L_iL_j}-P_{R_iR_j})}{min(P_L,P_R)}\f$ \n
 * where \f$P_L\f$ and \f$P_R\f$ are the node probabilities for the optimal split of node i into left and right nodes respectively,
 * \f$P_{L_iL_j}\f$ (\f$P_{R_iR_j}\f$ resp.) is the probability that both (optimal) node i and (surrogate) node j send an observation
 * to the Left (Right resp.). \n
 * We use best surrogate split, 2nd best surrogate split and so on until all data points with missing attributes in a node
 * have been sent to left/right child. If all possible surrogate splits are used up but some data points are still to be
 * assigned left/right child, majority rule is used, ie. the data points are assigned the child where majority of data points
 * have gone from the node. \n
 * cf. http://pic.dhe.ibm.com/infocenter/spssstat/v20r0m0/index.jsp?topic=%2Fcom.ibm.spss.statistics.help%2Falg_tree-cart.htm
 */
class CARTree : public RandomMixin<FeatureImportanceTree<CARTreeNodeData>>
{
public:
	/** default constructor */
	CARTree();

	/** constructor
	 * @param attribute_types type of each predictive attribute (true for nominal, false for ordinal/continuous)
	 * @param prob_type machine problem type - PT_MULTICLASS or PT_REGRESSION
	 */
	CARTree(SGVector<bool> attribute_types, EProblemType prob_type=PT_MULTICLASS);

	/** constructor - to be used while using cross-validation pruning
	 * @param attribute_types type of each predictive attribute (true for nominal, false for ordinal/continuous)
	 * @param prob_type machine problem type - PT_MULTICLASS or PT_REGRESSION
	 * @param num_folds number of subsets used in cross-valiation
	 * @param cv_prune - whether to use cross-validation pruning
	 */
	CARTree(SGVector<bool> attribute_types, EProblemType prob_type, int32_t num_folds, bool cv_prune);

	/** destructor */
	~CARTree() override;

	/** get name
	 * @return class name CARTree
	 */
	const char* get_name() const override { return "CARTree"; }

	/** get problem type - multiclass classification or regression
	 * @return PT_MULTICLASS or PT_REGRESSION
	 */
	EProblemType get_machine_problem_type() const override { return m_mode; }

	/** set problem type - multiclass classification or regression
	 * @param mode EProblemType PT_MULTICLASS or PT_REGRESSION
	 */
	void set_machine_problem_type(EProblemType mode);

	/** whether labels supplied are valid for current problem type
	 * @param lab labels supplied
	 * @return true for valid labels, false for invalid labels
	 */
	bool is_label_valid(std::shared_ptr<Labels> lab) const override;

	/** classify data using Classification Tree
	 * @param data data to be classified
	 * @return MulticlassLabels corresponding to labels of various test vectors
	 */
	std::shared_ptr<MulticlassLabels> apply_multiclass(std::shared_ptr<Features> data=NULL) override;

	/** Get regression labels using Regression Tree
	 * @param data data whose regression output is needed
	 * @return Regression output for various test vectors
	 */
	std::shared_ptr<RegressionLabels> apply_regression(std::shared_ptr<Features> data=NULL) override;

	/** uses test dataset to choose best pruned subtree
	 *
	 * @param feats test data to be used
	 * @param gnd_truth test labels
	 * @param weights weights of data points
	 */
	void prune_using_test_dataset(const std::shared_ptr<DenseFeatures<float64_t>>& feats, const std::shared_ptr<Labels>& gnd_truth, SGVector<float64_t> weights=SGVector<float64_t>());

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

	/** get number of subsets used for cross validation
	 *
	 * @return number of folds used in cross validation
	 */
	int32_t get_num_folds() const;

	/** set number of subsets for cross validation
	 *
	 * @param folds number of folds used in cross validation
	 */
	void set_num_folds(int32_t folds);

	/** get max allowed tree depth
	 *
	 * @return max allowed tree depth
	 */
	int32_t get_max_depth() const;

	/** set max allowed tree depth
	 *
	 * @param depth max allowed tree depth
	 */
	void set_max_depth(int32_t depth);

	/** get min allowed node size
	 *
	 * @return min allowed node size
	 */
	int32_t get_min_node_size() const;

	/** set min allowed node size
	 *
	 * @param nsize min allowed node size
	 */
	void set_min_node_size(int32_t nsize);

	/** Set cross validation pruning parameter
	 *
	 * @param cv_pruning allow CV pruning
	 */
	void set_cv_pruning(bool cv_pruning)
	{
		m_apply_cv_pruning = cv_pruning;
	}

	/** get label epsilon
	 *
	 * @return equality range for regression labels
	 */
	float64_t get_label_epsilon() { return m_label_epsilon; }

	/** set label epsilon
	 *
	 * @param epsilon equality range for regression labels
	 */
	void set_label_epsilon(float64_t epsilon);

	void pre_sort_features(const std::shared_ptr<Features>& data, SGMatrix<float64_t>& sorted_feats, SGMatrix<index_t>& sorted_indices);

	void set_sorted_features(SGMatrix<float64_t>& sorted_feats, SGMatrix<index_t>& sorted_indices);

	/**return feature importance
	 * this way is the same as sklearn
	 */
	SGVector<float64_t> get_feature_importance();

protected:
	/** train machine - build CART from training data
	 * @param data training data
	 * @return true
	 */
	bool train_machine(const std::shared_ptr<Features>& data, const std::shared_ptr<Labels>& labs) override;

	/** CARTtrain - recursive CART training method
	 *
	 * @param data training data
	 * @param weights vector of weights of data points
	 * @param labels labels of data points
	 * @param level current tree depth
	 * @return pointer to the root of the CART subtree
	 */
	virtual std::shared_ptr<BinaryTreeMachineNode<CARTreeNodeData>> CARTtrain(std::shared_ptr<DenseFeatures<float64_t>> data, const SGVector<float64_t>& weights, std::shared_ptr<DenseLabels> labels, int32_t level);

	/** modify labels for compute_best_attribute
	 *
	 * @param labels_vec labels vector
	 * @param n_ulabels stores number of unique labels
	 * @return unique labels
	 */
	SGVector<float64_t> get_unique_labels(const SGVector<float64_t>& labels_vec, index_t &n_ulabels) const;

	/** computes best attribute for CARTtrain
	 *
	 * @param mat data matrix
	 * @param weights data weights
	 * @param labels_vec data labels
	 * @param left stores feature values for left transition
	 * @param right stores feature values for right transition
	 * @param is_left_final stores which feature vectors go to the left child
	 * @param num_missing number of missing attributes
	 * @param count_left stores number of feature values for left transition
	 * @param count_right stores number of feature values for right transition
	 * @param impurity impurity of current node
	 * @return index to the best attribute
	 */
	virtual index_t compute_best_attribute(
		const SGMatrix<float64_t>& mat, const SGVector<float64_t>& weights,
		std::shared_ptr<DenseLabels> labels, SGVector<float64_t>& left,
		SGVector<float64_t>& right, SGVector<bool>& is_left_final,
		index_t& num_missing, index_t& count_left, index_t& count_right,
		float64_t& impurity, index_t subset_size = 0,
		const SGVector<index_t>& active_indices = SGVector<index_t>());

	/** handles missing values through surrogate splits
	 *
	 * @param data training data matrix
	 * @param weights vector of weights of data points
	 * @param nm_left whether a data point is put into left child (available for only data points with non-missing attribute attr)
	 * @param attr best attribute chosen for split
	 * @return vector denoting whether a data point goes to left child for all data points including ones with missing attributes
	 */
	SGVector<bool> surrogate_split(SGMatrix<float64_t> data, SGVector<float64_t> weights, SGVector<bool> nm_left, int32_t attr) const;


	/** handles missing values for a chosen continuous surrogate attribute
	 *
	 * @param m training data matrix
	 * @param missing_vecs column indices of vectors with missing attribute in data matrix
	 * @param association_index stores the final lambda values used to address members of missing_vecs
	 * @param intersect_vecs column indices of vectors with known values for the best attribute as well as the chosen surrogate
	 * @param is_left whether a vector goes into left child
	 * @param weights weights of training data vectors
	 * @param p min(p_l,p_r) in the lambda formula
	 * @param attr surrogate attribute chosen for split
	 * @return vector denoting whether a data point goes to left child for all data points including ones with missing attributes
	 */
	void handle_missing_vecs_for_continuous_surrogate(SGMatrix<float64_t> m, const std::vector<index_t>& missing_vecs,
		std::vector<float64_t>& association_index, std::vector<index_t>& intersect_vecs,
		SGVector<bool> is_left, SGVector<float64_t> weights, float64_t p, index_t attr) const;

	/** handles missing values for a chosen nominal surrogate attribute
	 *
	 * @param m training data matrix
	 * @param missing_vecs column indices of vectors with missing attribute in data matrix
	 * @param association_index stores the final lambda values used to address members of missing_vecs
	 * @param intersect_vecs column indices of vectors with known values for the best attribute as well as the chosen surrogate
	 * @param is_left whether a vector goes into left child
	 * @param weights weights of training data vectors
	 * @param p min(p_l,p_r) in the lambda formula
	 * @param attr surrogate attribute chosen for split
	 * @return vector denoting whether a data point goes to left child for all data points including ones with missing attributes
	 */
	void handle_missing_vecs_for_nominal_surrogate(SGMatrix<float64_t> m, const std::vector<index_t>& missing_vecs,
		std::vector<float64_t>& association_index, const std::vector<index_t>& intersect_vecs,
		SGVector<bool> is_left, SGVector<float64_t> weights, float64_t p, index_t attr) const;

	/** returns gain in regression case
	 *
	 * @param wleft left child weight distribution
	 * @param wright right child weights distribution
	 * @param wtotal weight distribution in current node
	 * @param labels regression labels
	 * @return least squared deviation gain achieved after spliting the node
	 */
	float64_t gain(
		const SGVector<float64_t>& wleft, const SGVector<float64_t>& wright,
		const SGVector<float64_t>& wtotal, const SGVector<float64_t>& feats,
		float64_t& impurity) const;

	/** returns gain in Gini impurity measure
	 *
	 * @param wleft left child label distribution
	 * @param wright right child label distribution
	 * @param wtotal label distribution in current node
	 * @return Gini gain achieved after spliting the node
	 */
	float64_t gain(
		const SGVector<float64_t>& wleft, const SGVector<float64_t>& wright,
		const SGVector<float64_t>& wtotal, float64_t& impurity) const;

	/** returns Gini impurity of a node
	 *
	 * @param weighted_lab_classes vector of weights associated with various labels
	 * @param total_weight stores the total weight of all classes
	 * @return Gini index of the node
	 */
	float64_t gini_impurity_index(const SGVector<float64_t>& weighted_lab_classes, float64_t &total_weight) const;

	/** returns least squares deviation
	 *
	 * @param labels regression labels
	 * @param weights weights of regression data points
	 * @param total_weight stores sum of weights in weights vector
	 * @return least squares deviation of the data
	 */
	float64_t least_squares_deviation(const SGVector<float64_t>& labels, const SGVector<float64_t>& weights, float64_t &total_weight) const;

	/** uses current subtree to classify/regress data
	 *
	 * @param feats data to be classified/regressed
	 * @param current root of current subtree
	 * @return classification/regression labels of input data
	 */
	std::shared_ptr<Labels> apply_from_current_node(const std::shared_ptr<DenseFeatures<float64_t>>& feats, const std::shared_ptr<bnode_t>& current);

	/** prune by cross validation
	 *
	 * @param data training data
	 * @param folds the integer V for V-fold cross validation
	 */
	void prune_by_cross_validation(const std::shared_ptr<DenseFeatures<float64_t>>& data, const std::shared_ptr<Labels>& labs, int32_t folds);

	/** computes error in classification/regression
	 * for classification it eveluates weight_missclassified/total_weight
	 * for regression it evaluates weighted sum of squared error/total_weight
	 *
	 * @param labels the labels whose error needs to be calculated
	 * @param reference actual labels against which test labels are compared
	 * @param weights weights associated with the labels
	 * @return error evaluated
	 */
	float64_t compute_error(const std::shared_ptr<Labels>& labels, const std::shared_ptr<Labels>& reference, SGVector<float64_t> weights) const;

	/** cost-complexity pruning
	 *
	 * @param tree the tree to be pruned
	 * @return array of pruned trees
	 */
	std::vector<std::shared_ptr<bnode_t>> prune_tree(const std::shared_ptr<TreeMachine<CARTreeNodeData>>& tree);

	/** recursively finds alpha corresponding to weakest link(s)
	 *
	 * @param node the root of subtree whose weakest link it finds
	 * @return alpha value corresponding to the weakest link in subtree
	 */
	float64_t find_weakest_alpha(const std::shared_ptr<bnode_t>& node) const;

	/** recursively cuts weakest link(s) in a tree
	 *
	 * @param node the root of subtree whose weakest link it cuts
	 * @param alpha alpha value corresponding to weakest link
	 */
	void cut_weakest_link(const std::shared_ptr<bnode_t>& node, float64_t alpha);

	/** recursively forms base case $ft_1$f tree from $ft_max$f during pruning
	 *
	 * @param node the root of current subtree
	 */
	void form_t1(const std::shared_ptr<bnode_t>& node);

	/** initializes members of class */
	void init();

	void set_machine_problem_type(const std::shared_ptr<Labels>& labs)
	{
		if (labs->get_label_type()==LT_MULTICLASS)
			set_machine_problem_type(PT_MULTICLASS);
		else if (labs->get_label_type()==LT_REGRESSION)
			set_machine_problem_type(PT_REGRESSION);
		else
			error("label type supplied is not supported");
	}
public:
	/** denotes that a feature in a vector is missing MISSING = NOT_A_NUMBER */
	static const float64_t MISSING;

	/** min gain for splitting to be allowed */
	static const float64_t MIN_SPLIT_GAIN;

	/** equality epsilon */
	static const float64_t EQ_DELTA;

protected:
	/** Returns whether the type of various feature dimensions are specified
	 * using is_nominal_feature
	 */
	bool types_set();

	/** Returns whether weights are set */
	bool weights_set();

	/** equality range for regression labels */
	float64_t m_label_epsilon;

	/** vector depicting whether various feature dimensions are nominal or not **/
	SGVector<bool> m_nominal;

	/** weights of samples in training set **/
	SGVector<float64_t> m_weights;

	/** sorted transposed features */
	SGMatrix<float64_t> m_sorted_features;

	/** sorted indices */
	SGMatrix<index_t> m_sorted_indices;

	/** If pre sorted features are used in train */
	bool m_pre_sort;

	/** flag indicating whether cross validation pruning has to be applied or not - false by default **/
	bool m_apply_cv_pruning;

	/** V in V-fold cross validation - 5 by default **/
	int32_t m_folds;

	/** Problem type : PT_MULTICLASS or PT_REGRESSION **/
	EProblemType m_mode;

	/** stores \f$\alpha_k\f$ values evaluated in cost-complexity pruning **/
	std::vector<float64_t> m_alphas;

	/** max allowed depth of tree **/
	int32_t m_max_depth;

	/** minimum number of feature vectors required in a node **/
	int32_t m_min_node_size;
};
} /* namespace shogun */

#endif /* _CARTREE_H__ */
