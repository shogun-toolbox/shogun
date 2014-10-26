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

#ifndef _RANDOMFOREST_H__
#define _RANDOMFOREST_H__

#include <shogun/lib/config.h>
#include <shogun/machine/BaggingMachine.h>

namespace shogun
{

/** @brief This class implements the Random Forests algorithm. In Random Forests algorithm, we train a number of randomized CART trees 
 * (see class CRandomCARTree) using the supplied training data. The number of trees to be trained is a parameter (called number of bags)  
 * controlled by the user. Test feature vectors are classified/regressed by combining the outputs of all these trained candidate trees using a
 * combination rule (see class CCombinationRule). The feature for calculating out-of-box error is also provided to help determine the  
 * appropriate number of bags. The evaluatin criteria for calculating this out-of-box error is specified by the user (see class CEvaluation).
 */
class CRandomForest : public CBaggingMachine
{
public:
	/** constructor */
	CRandomForest();

	/** constructor
	 *
	 * @param num_rand_feats number of attributes chosen randomly during node split in candidate trees 
	 * @param num_bags number of trees in forest 
	 */
	CRandomForest(int32_t num_rand_feats, int32_t num_bags=10);

	/** constructor
	 *
	 * @param features training features
	 * @param labels training labels
	 * @param num_bags number of trees in forest
	 * @param num_rand_feats number of attributes chosen randomly during node split in candidate trees  
	 */
	CRandomForest(CFeatures* features, CLabels* labels, int32_t num_bags=10, int32_t num_rand_feats=0);

	/** constructor
	 *
	 * @param features training features
	 * @param labels training labels
	 * @param weights weights of training feature vectors
	 * @param num_bags number of trees in forest
	 * @param num_rand_feats number of attributes chosen randomly during node split in candidate trees
	 */
	CRandomForest(CFeatures* features, CLabels* labels, SGVector<float64_t> weights, int32_t num_bags=10, int32_t num_rand_feats=0);

	/** destructor */
	virtual ~CRandomForest();

	/** get name
	 *
	 * @return RandomForest
	 */
	virtual const char* get_name() const { return "RandomForest"; }

	/** machine is set to modified CART(RandomCART) and cannot be changed
	 *
	 * @param machine the machine to use for bagging
	 */
	virtual void set_machine(CMachine* machine);

	/** set weights
	 *
	 * @param weights of training feature vectors
	 */
	void set_weights(SGVector<float64_t> weights);

	/** get weights
	 *
	 * @return weights of training feature vectors
	 */
	SGVector<float64_t> get_weights() const;

	/** set feature types of various features
	 * 
	 * @param ft bool vector true for nominal feature false for continuous feature type 
	 */
	void set_feature_types(SGVector<bool> ft);

	/** get feature types of various features
	 *
	 * @return bool vector - true for nominal feature false for continuous feature type 
	 */
	SGVector<bool> get_feature_types() const;

	/** get problem type - multiclass classification or regression
	 *
	 * @return PT_MULTICLASS or PT_REGRESSION
	 */
	virtual EProblemType get_machine_problem_type() const; 

	/** set problem type - multiclass classification or regression
	 *
	 * @param mode EProblemType PT_MULTICLASS or PT_REGRESSION
	 */
	void set_machine_problem_type(EProblemType mode);

	/** set number of random features to be chosen during node splits
	 *
	 * @param rand_featsize number of randomly chosen features during each node split
	 */
	void set_num_random_features(int32_t rand_featsize);

	/** get number of random features to be chosen during node splits
	 *
	 * @return number of randomly chosen features during each node split
	 */
	int32_t get_num_random_features() const;

protected:
	/** sets parameters of CARTree - sets machine labels and weights here
	 *
	 * @param m machine
	 * @param idx indices of training vectors chosen in current bag
	 */
	virtual void set_machine_parameters(CMachine* m, SGVector<index_t> idx);

private:
	/** initialize parameters */
	void init();

private:
	/** weights */
	SGVector<float64_t> m_weights;

};
} /* namespace shogun */
#endif /* _RANDOMFOREST_H__ */
