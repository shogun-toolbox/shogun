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


#ifndef _CHAIDTree_H__
#define _CHAIDTree_H__

#include <shogun/lib/config.h>

#include <shogun/multiclass/tree/TreeMachine.h>
#include <shogun/multiclass/tree/CHAIDTreeNodeData.h>
#include <shogun/features/DenseFeatures.h>

namespace shogun
{

/** @brief This class implements the CHAID algorithm proposed by Kass (1980) for decision tree learning.
 * CHAID consists of three steps: merging, splitting and stopping. A tree is grown by repeatedly using these 
 * three steps on each node starting from the root node. CHAID accepts nominal or ordinal categorical predictors
 * only. If predictors are continuous, they have to be transformed into ordinal predictors before tree growing.
 * \n \n
 * CONVERTING CONTINUOUS PREDICTORS TO ORDINAL : \n
 * Continuous predictors are converted to ordinal by binning. The number of bins (K) has to be supplied by the user.
 * Given K, a predictor is split in such a way that all the bins get the same number (more or less) of distinct 
 * predictor values. The maximum feature value in each bin is used as a breakpoint.
 * \n \n
 * MERGING : \n
 * During the merging step, allowable pairs of categories of a predictor are evaluated for similarity. If the similarity
 * of a pair is above a threshold, the categories constituting the pair are merged into a single category. The process is 
 * repeated until there is no pair left having high similarity between its categories. Similarity between categories is
 * evaluated using the p_value
 * \n \n
 * SPLITTING : \n
 * The splitting step selects which predictor to be used to best split the node. Selection is accomplished by comparing
 * the adjusted p_value associated with each predictor. The predictor that has the smallest adjusted p_value is chosen
 * for splitting the node.
 * \n \n
 * STOPPING : \n
 * The tree growing process stops if any of the following conditions is satisfied : \n
 * 1. If a node becomes pure; that is, all cases in a node have identical values of the dependent variable, 
 * the node will not be split. \n
 * 2. If all cases in a node have identical values for each predictor, the node will not be split. \n
 * 3. If the current tree depth reaches the user specified maximum tree depth limit value, the tree growing process will stop. \n
 * 4. If the size of a node is less than the user-specified minimum node size value, the node will not be split.
 * \n \n
 * p_value CALCULATIONS FOR NOMINAL DEPENDENT VARIABLE: \n
 * If the dependent variable is nominal categorical, a contingency (or count) table is formed using classes of Y as 
 * columns and categories of the predictor X as rows. The p_value is computed using the entries of this table and 
 * Pearson chi-squared statistic, For more details, please see : 
 * http://pic.dhe.ibm.com/infocenter/spssstat/v20r0m0/index.jsp?topic=%2Fcom.ibm.spss.statistics.help%2Falg_tree-chaid_pvalue_categorical.htm 
 * \n \n
 * p_value CALCULATIONS FOR ORDINAL DEPENDENT VARIABLE: \n
 * If the dependent variable Y is categorical ordinal, the null hypothesis of independence of X and Y is tested
 * against the row effects model, with the rows being the categories of X and columns the classes of Y. Again Pearson chi-
 * squared statistic is used (like nominal case) but two sets of expected cell frequencies are calculated. For more details :
 * http://pic.dhe.ibm.com/infocenter/spssstat/v20r0m0/index.jsp?topic=%2Fcom.ibm.spss.statistics.help%2Falg_tree-chaid_pvalue_ordinal.htm
 * \n \n
 * p_value CALCULATIONS FOR CONTINUOUS DEPENDENT VARIABLE: \n
 * If the dependent variable Y is continuous, an ANOVA F test is performed that tests if the means of Y for 
 * different categories of X are the same. For more details please see : 
 * http://pic.dhe.ibm.com/infocenter/spssstat/v20r0m0/index.jsp?topic=%2Fcom.ibm.spss.statistics.help%2Falg_tree-chaid_pvalue_scale.htm 
 */
class CCHAIDTree : public CTreeMachine<CHAIDTreeNodeData>
{
public:
	/** default constructor */
	CCHAIDTree();

	/** constructor
	 * @param dependent_vartype feature type for dependent variable (0-nominal, 1-ordinal or 2-continuous)  
	 */
	CCHAIDTree(int32_t dependent_vartype);

	/** destructor */
	virtual ~CCHAIDTree();

	/** get name
	 * @return class name CHAIDTree
	 */
	virtual const char* get_name() const { return "CHAIDTree"; }

	/** get problem type - multiclass classification or regression
	 * @return PT_MULTICLASS or PT_REGRESSION
	 */
	virtual EProblemType get_machine_problem_type() const;

	/** whether labels supplied are valid for current problem type
	 * @param lab labels supplied
	 * @return true for valid labels, false for invalid labels
	 */
	virtual bool is_label_valid(CLabels* lab) const;

	/** classify data using Classification Tree
	 * @param data data to be classified
	 * @return MulticlassLabels corresponding to labels of various test vectors 
	 */
	virtual CMulticlassLabels* apply_multiclass(CFeatures* data=NULL);

	/** Get regression labels using Regression Tree
	 * @param data data whose regression output is needed 
	 * @return Regression output for various test vectors 
	 */
	virtual CRegressionLabels* apply_regression(CFeatures* data=NULL);

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
	 * @param ft vector with feature types : 0-nominal, 1-ordinal or 2-continuous 
	 */
	void set_feature_types(SGVector<int32_t> ft);

	/** get feature types of various features
	 * @return vector with feature types : 0-nominal, 1-ordinal or 2-continuous
	 */
	SGVector<int32_t> get_feature_types() const;

	/** clear feature types of various features */
	void clear_feature_types();

	/** set dependent variable type : 0 for nominal, 1 for ordinal and 2 for continuous 
	 * @param var integer corresponding to the dependent variable type
	 */
	void set_dependent_vartype(int32_t var); 

	/** get dependent variable type : 0 for nominal, 1 for ordinal and 2 for continuous 
	 * @return integer corresponding to the dependent variable type
	 */
	int32_t get_dependent_vartype() const { return m_dependent_vartype; }

	/** set max tree depth
	 * @param d max tree depth
	 */
	void set_max_tree_depth(int32_t d) { m_max_tree_depth=d; }

	/** get max tree depth
	 * @return d max tree depth
	 */
	int32_t get_specified_max_tree_depth() const { return m_max_tree_depth; }

	/** set minimum node size
	 * @param size min node size
	 */
	void set_min_node_size(int32_t size) { m_min_node_size=size; }

	/** get minimum node size
	 * @return size min node size
	 */
	int32_t get_min_node_size() const { return m_min_node_size; }

	/** set alpha_merge
	 * @param a alpha_merge
	 */
	void set_alpha_merge(float64_t a) { m_alpha_merge=a; }

	/** get alpha_merge
	 * @return a alpha_merge
	 */
	float64_t get_alpha_merge() const { return m_alpha_merge; }

	/** set alpha_split
	 * @param a alpha_split
	 */
	void set_alpha_split(float64_t a) { m_alpha_split=a; }

	/** get alpha_split
	 * @return a alpha_split
	 */
	float64_t get_alpha_split() const { return m_alpha_split; }

	/** set number of breakpoints
	 * @param b number of breakpoints
	 */
	void set_num_breakpoints(int32_t b) { m_num_breakpoints=b; }

	/** get number of breakpoints
	 * @return number of breakpoints
	 */
	float64_t get_num_breakpoints() const { return m_num_breakpoints; }

protected:
	/** train machine - build CHAID from training data
	 * @param data training data
	 * @return true
	 */
	virtual bool train_machine(CFeatures* data=NULL);

private:
	/** CHAIDtrain - recursive CHAID training method
	 *
	 * @param data training data
	 * @param weights vector of weights of data points
	 * @param labels labels of data points
	 * @param level current depth of tree
	 * @return pointer to the root of the CHAID subtree
	 */
	CTreeMachineNode<CHAIDTreeNodeData>* CHAIDtrain(CFeatures* data, SGVector<float64_t> weights, CLabels* labels, int32_t level);

	/** executes merge step of the tree growing process for ordinal features
	 *
	 * @param feats feature values for chosen predictive attribute
	 * @param labels data labels
	 * @param weights data weights
	 * @param pv stores p_value
	 * @return vector with category labels for each unique value of the chosen attribute
	 */
	SGVector<int32_t> merge_categories_ordinal(SGVector<float64_t> feats, SGVector<float64_t> labels, 
								SGVector<float64_t> weights, float64_t &pv);

	/** executes merge step of the tree growing process for nominal features
	 *
	 * @param feats feature values for chosen predictive attribute
	 * @param labels data labels
	 * @param weights data weights
	 * @param pv stores p_value
	 * @return vector with category labels for each unique value of the chosen attribute
	 */
	SGVector<int32_t> merge_categories_nominal(SGVector<float64_t> feats, SGVector<float64_t> labels, 
								SGVector<float64_t> weights, float64_t &pv);

	/** applies the decision tree learnt to test data
	 *
	 * @param data test data to be classified/regressed
	 * @return classification/regression labels of input data
	 */
	CLabels* apply_tree(CFeatures* data);

	/** uses current subtree to classify/regress data
	 *
	 * @param fmat data matrix to be classified/regressed
	 * @param current root of current subtree
	 * @return classification/regression labels of input data
	 */
	CLabels* apply_from_current_node(SGMatrix<float64_t> fmat, node_t* current);

	/** handles missing values category for ordinal feature type
	 *
	 * @param cat category vector
	 * @param feats feature vector represented in terms of categories
	 * @param labels labels associated with features
	 * @param weights weights associated with features
	 * @return whether missing values category is merged 
	 */
	bool handle_missing_ordinal(SGVector<int32_t> cat, SGVector<float64_t> feats, SGVector<float64_t> labels, SGVector<float64_t> weights);

	/** calculates adjusted p_value using Bonferroni adjustments
	 * 
	 * @param p_value unadjusted p_value 
	 * @param inum_cat number of categories of a predictor before merging
	 * @param fnum_cat number of categories of a predictor after merging
	 * @param ft feature type : 0 for nominal 1 for ordinal
	 * @param is_missing whether missing values are present. Affects adjustment factor for ordinal feature type
	 * @return adjusted p_value
	 */
	float64_t adjusted_p_value(float64_t p_value, int32_t inum_cat, int32_t fnum_cat, int32_t ft, bool is_missing);

	/** calculates unadjusted p_value
	 *
	 * @param feat chosen attribute values of all data vectors 
	 * @param labels labels associated with data
	 * @param weights weights associated with data
	 * @return p_value of the data
	 */
	float64_t p_value(SGVector<float64_t> feat, SGVector<float64_t> labels, SGVector<float64_t> weights);

	/** calculates ANOVA F-statistic
	 *
	 * @param feat chosen attribute values of all data vectors 
	 * @param labels labels associated with data
	 * @param weights weights associated with data
	 * @param r stores unique feature categories
	 * @return  ANOVA F-statistic of the data
	 */
	float64_t anova_f_statistic(SGVector<float64_t> feat, SGVector<float64_t> labels, SGVector<float64_t> weights, int32_t &r);

	/** calculates likelihood ratio statistic
	 *
	 * @param feat chosen attribute values of all data vectors 
	 * @param labels labels associated with data
	 * @param weights weights associated with data
	 * @param r stores number of rows in contingency table
	 * @param c stores number of columns in contingency table
	 * @return  likelihood ratio of the data
	 */
	float64_t likelihood_ratio_statistic(SGVector<float64_t> feat, SGVector<float64_t> labels, SGVector<float64_t> weights,
													 int32_t &r, int32_t &c);

	/** calculates Pearson's Chi-squared statistic
	 *
	 * @param feat chosen attribute values of all data vectors 
	 * @param labels labels associated with data
	 * @param weights weights associated with data
	 * @param r stores number of rows in contingency table
	 * @param c stores number of columns in contingency table
	 * @return  Pearson's Chi-squared statistic of the data
	 */
	float64_t pchi2_statistic(SGVector<float64_t> feat, SGVector<float64_t> labels, SGVector<float64_t> weights, int32_t &r, int32_t &c);

	/** calculates hypothesis under row effects model
	 *
	 * @param ct contingency table 
	 * @param wt weight table [Note : the weight table is modified by the method]
	 * @param score score of class of columns [Note : the score vector is modified by the method]
	 * @return  matrix containing estimated cell frequencies 
	 */
	SGMatrix<float64_t> expected_cf_row_effects_model(SGMatrix<int32_t> ct, SGMatrix<float64_t> wt, SGVector<float64_t> score);

	/** calculates null hypothesis of independence
	 *
	 * @param ct contingency table 
	 * @param wt weight table [Note : the weight table is modified by the method]
	 * @return  matrix containing estimated cell frequencies 
	 */
	SGMatrix<float64_t> expected_cf_indep_model(SGMatrix<int32_t> ct, SGMatrix<float64_t> wt);

	/** calculates node label for continuous dependent variable using sum of squared deviation
	 * @param lab labels
	 * @param weights associated weights
	 * @param mean stores the mean label
	 * @return sum of squared deviation 
	 */
	float64_t sum_of_squared_deviation(SGVector<float64_t> lab, SGVector<float64_t> weights, float64_t &mean);

	/** converts continuous features to ordinal via binning. Forms conversion matrix m_cont_breakpoints.
	 * NOTE : This method changes data matrix. The continuous feature values
	 * are replaced with the actual feature values used.
	 * 
	 * @param featmat feature matrix
	 * @return whether data matrix is updated
	 */
	bool continuous_to_ordinal(SGMatrix<float64_t> featmat);

	/** initializes members of class */
	void init();

public:
	/** denotes that a feature in a vector is missing MISSING = MAX_REAL_NUMBER */
	static const float64_t MISSING;

private:
	/** vector depicting whether various feature dimensions are nominal(0) ordinal(1) or continuous(2) **/
	SGVector<int32_t> m_feature_types;

	/** weights of samples in training set **/
	SGVector<float64_t> m_weights;

	/** whether weights are set */
	bool m_weights_set;

	/** whether dependent variable is nominal(0), ordinal(1) or continuous(2) */
	int32_t m_dependent_vartype;

	/** max allowed tree depth */
	int32_t m_max_tree_depth;

	/** min allowed node size */
	int32_t m_min_node_size;

	/** alpha_merge value */
	float64_t m_alpha_merge;

	/** alpha_split value */
	float64_t m_alpha_split;

	/** continuous to ordinal conversion matrix */
	SGMatrix<float64_t> m_cont_breakpoints;

	/** number of breakpoints for continuous to ordinal conversion */
	int32_t m_num_breakpoints;

};
} /* namespace shogun */

#endif /* _CHAIDTree_H__ */
