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
 * of a pair is above threshold, the categories constituting the pair are merged into a single category. The process is 
 * repeated until there is no pair left having high similarity between its categories. Similarity between categories is
 * evaluated using p-value
 * \n \n
 * SPLITTING : \n
 * The splitting step selects which predictor to be used to best split the node. Selection is accomplished by comparing
 * the adjusted p-value associated with each predictor. The predictor that has the smallest adjusted p-value is chosen
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
 * p-VALUE CALCULATIONS FOR NOMINAL DEPENDENT VARIABLE: \n
 * If dependent variable is nominal categorical, a contingency (or count) table is formed using classes of Y as 
 * columns and categories of the predictor X as rows. The p-value is computed using the entries of this table and 
 * Pearson chi-squared statistic, For more details, please see : 
 * http://pic.dhe.ibm.com/infocenter/spssstat/v20r0m0/index.jsp?topic=%2Fcom.ibm.spss.statistics.help%2Falg_tree-chaid_pvalue_categorical.htm 
 * \n \n
 * p-VALUE CALCULATIONS FOR ORDINAL DEPENDENT VARIABLE: \n
 * If the dependent variable Y is categorical ordinal, the null hypothesis of independence of X and Y is tested
 * against the row effects model, with the rows being the categories of X and columns the classes of Y. Again Pearson chi-
 * squared statistic is used (like nominal case) but two sets of expected cell frequencies are calculated. For more details :
 * http://pic.dhe.ibm.com/infocenter/spssstat/v20r0m0/index.jsp?topic=%2Fcom.ibm.spss.statistics.help%2Falg_tree-chaid_pvalue_ordinal.htm
 * \n \n
 * p-VALUE CALCULATIONS FOR CONTINUOUS DEPENDENT VARIABLE: \n
 * If the dependent variable Y is continuous, an ANOVA F test is performed that tests if the means of Y for 
 * different categories of X are the same. For more details please see : 
 * http://pic.dhe.ibm.com/infocenter/spssstat/v20r0m0/index.jsp?topic=%2Fcom.ibm.spss.statistics.help%2Falg_tree-chaid_pvalue_scale.htm 
 */
class CCHAIDTree : public CTreeMachine<CHAIDTreeNodeData>
{
public:
	/** constructor */
	CCHAIDTree();

	/** destructor */
	virtual ~CCHAIDTree();

	/** get name
	 * @return class name CHAIDTree
	 */
	virtual const char* get_name() const { return "CHAIDTree"; }

	/** get problem type - multiclass classification or regression
	 * @return PT_MULTICLASS or PT_REGRESSION
	 */
	virtual EProblemType get_machine_problem_type() const { return m_mode; } 

	/** set problem type - multiclass classification or regression
	 * @param mode EProblemType PT_MULTICLASS or PT_REGRESSION
	 */
	void set_machine_problem_type(EProblemType mode);

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

	/** set feature types of various features
	 * @return vector with feature types : 0-nominal, 1-ordinal or 2-continuous
	 */
	SGVector<int32_t> get_feature_types() const;

	/** clear feature types of various features */
	void clear_feature_types();

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
	 * @return pointer to the root of the CHAID subtree
	 */
	CTreeMachineNode<CHAIDTreeNodeData>* CHAIDtrain(CFeatures* data, SGVector<float64_t> weights, CLabels* labels);

	/** uses current subtree to classify/regress data
	 *
	 * @param feats data to be classified/regressed
	 * @param current root of current subtree
	 * @return classification/regression labels of input data
	 */
	CLabels* apply_from_current_node(CDenseFeatures<float64_t>* feats, node_t* current);

	/** initializes members of class */
	void init();

public:
	/** denotes that a feature in a vector is missing MISSING = NOT_A_NUMBER */
	static const float64_t MISSING;

private:
	/** vector depicting whether various feature dimensions are nominal ordinal or continuous **/
	SGVector<int32_t> m_feature_types;

	/** weights of samples in training set **/
	SGVector<float64_t> m_weights;

	/** Problem type : PT_MULTICLASS or PT_REGRESSION **/
	EProblemType m_mode;

};
} /* namespace shogun */

#endif /* _CHAIDTree_H__ */
