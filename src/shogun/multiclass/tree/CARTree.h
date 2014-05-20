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

#include <shogun/lib/config.h>

#include <shogun/multiclass/tree/TreeMachine.h>
#include <shogun/multiclass/tree/CARTreeNodeData.h>
#include <shogun/features/DenseFeatures.h>

namespace shogun
{

/** @brief
 */
class CCARTree : public CTreeMachine<CARTreeNodeData>
{
public:
	/** constructor */
	CCARTree();

	/** destructor */
	virtual ~CCARTree();

	/** get name
	 * @return class name CARTree
	 */
	virtual const char* get_name() const { return "CARTree"; }

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
	/** train machine - build CART from training data
	 * @param data training data
	 * @return true
	 */
	virtual bool train_machine(CFeatures* data=NULL);

private:
	/** CARTtrain - recursive CART training method
	 *
	 * @param data training data
	 * @param weights vector of weights of data points
	 * @param labels labels of data points
	 * @return pointer to the root of the CART subtree
	 */
	CBinaryTreeMachineNode<CARTreeNodeData>* CARTtrain(CFeatures* data, SGVector<float64_t> weights, CLabels* labels);

	/** returns gain in gini impurity measure
	 *
	 * @param labels labels of data in the node
	 * @param weights weights of the data points in the node
	 * @param is_left whether a data point is proposed to be moved to left child node
	 * @return gini gain acheived after spliting the node
	 */
	float64_t gain(CLabels* labels, SGVector<float64_t> weights, SGVector<bool> is_left);

	/** returns gini impurity of a node
	 * 
	 * @param labels labels of data in the node
	 * @param weights weights of the data points in the node
	 * @return gini index of the node
	 */
	float64_t gini_impurity_index(CMulticlassLabels* labels, SGVector<float64_t> weights);

	/** returns least squares deviation
	 * 
	 * @param labels regression labels
	 * @param weights weights of regression data points
	 */
	float64_t least_squares_deviation(CRegressionLabels* labels, SGVector<float64_t> weights);

	/** uses current subtree to classify/regress data
	 *
	 * @param feats data to be classified/regressed
	 * @param current root of current subtree
	 * @return classification/regression labels of input data
	 */
	CLabels* apply_from_current_node(CDenseFeatures<float64_t>* feats, bnode_t* current);

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

	/** flag storing whether the type of various feature dimensions are specified using is_nominal_feature **/
	bool m_types_set;

	/** flag storing whether weights of samples are specified using weights vector **/
	bool m_weights_set;

	/** Problem type : PT_MULTICLASS or PT_REGRESSION **/
	EProblemType m_mode;

};
} /* namespace shogun */

#endif /* _CARTREE_H__ */
