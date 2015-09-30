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

#ifndef _StochasticGBMachine_H__
#define _StochasticGBMachine_H__


#include <shogun/lib/config.h>

#include <shogun/machine/Machine.h>
#include <shogun/loss/LossFunction.h>
#include <shogun/features/DenseFeatures.h>

namespace shogun
{

/** @brief This class implements the stochastic gradient boosting algorithm for ensemble learning invented by Jerome H. Friedman. This class
 * works with a variety of loss functions like squared loss, exponential loss, Huber loss etc which can be accessed through Shogun's 
 * CLossFunction interface (cf. http://www.shogun-toolbox.org/doc/en/latest/classshogun_1_1CLossFunction.html). Additionally, it can create
 * an ensemble of any regressor class derived from the CMachine class (cf. http://www.shogun-toolbox.org/doc/en/latest/classshogun_1_1CMachine.html).
 * For one dimensional optimization, this class uses the backtracking linesearch accessed via Shogun's L-BFGS class.
 * A concise description of the algorithm implemented can be found in the following link : 
 * http://en.wikipedia.org/wiki/Gradient_boosting#Algorithm
 */
class CStochasticGBMachine : public CMachine
{
public:
	/** Constructor
	 *
	 * @param machine The class of machine which will constitute the ensemble 
	 * @param loss loss function
	 * @param num_iterations number of iterations of boosting
	 * @param subset_fraction fraction of trainining vectors to be chosen randomly w/o replacement
	 * @param learning_rate shrinkage factor
	 */
	CStochasticGBMachine(CMachine* machine=NULL, CLossFunction* loss=NULL, int32_t num_iterations=100, 
						float64_t learning_rate=1.0, float64_t subset_fraction=0.6);

	/** Destructor */
	virtual ~CStochasticGBMachine();

	/** get name
	 *
	 * @return StochasticGBMachine
	 */
	virtual const char* get_name() const { return "StochasticGBMachine"; }

	/** set machine
	 *
	 * @param machine machine
	 */
	void set_machine(CMachine* machine);

	/** get machine
	 *
	 * @return machine
	 */
	CMachine* get_machine() const;

	/** set loss function
	 *
	 * @param f loss function
	 */
	virtual void set_loss_function(CLossFunction* f);

	/** get loss function
	 *
	 * @return loss function
	 */
	virtual CLossFunction* get_loss_function() const;

	/** set number of iterations
	 *
	 * @param iter number of iterations
	 */
	void set_num_iterations(int32_t iter);

	/** get number of iterations
	 *
	 * @return number of iterations
	 */
	int32_t get_num_iterations() const;

	/** set subset fraction
	 *
	 * @param frac subset fraction (should lie between 0 and 1)
	 */
	void set_subset_fraction(float64_t frac);

	/** get subset fraction
	 *
	 * @return subset fraction
	 */
	float64_t get_subset_fraction() const;

	/** set learning rate
	 *
	 * @param lr learning rate
	 */
	void set_learning_rate(float64_t lr);

	/** get learning rate
	 *
	 * @return learning rate
	 */
	float64_t get_learning_rate() const;

	/** apply_regression
	 *
	 * @param data test data
	 * @return Regression labels
	 */
	virtual CRegressionLabels* apply_regression(CFeatures* data=NULL);

protected:
	/** train machine
	 *
	 * @param data training data
	 * @return true 
	 */
	virtual bool train_machine(CFeatures* data=NULL);

	/** compute gamma values
	 *
	 * @param f labels from the intermediate model
	 * @param hm labels from the newly trained base model
	 * @return gamma - the scalar weights given to individual weak learners in the ensemble model
	 */
	float64_t compute_multiplier(CRegressionLabels* f, CRegressionLabels* hm);

	/** train base model
	 *
	 * @param feats training data
	 * @param labels training labels
	 * @return trained base model
	 */
	CMachine* fit_model(CDenseFeatures<float64_t>* feats, CRegressionLabels* labels);

	/** compute pseudo_residuals
	 *
	 * @param inter_f intermediate boosted model labels for training data
	 * @return pseudo_residuals
	 */
	CRegressionLabels* compute_pseudo_residuals(CRegressionLabels* inter_f);

	/** add randomized subset to relevant parameters
	 *
	 * @param f training data
	 * @param interf intermediate boosted model labels for training data
	 */
	void apply_subset(CDenseFeatures<float64_t>* f, CLabels* interf);

	/** reset arrays of weak learners and gamma values */
	void initialize_learners();

	/** apply lbfgs to get gamma
	 *
	 * @param instance stores parameters to be passed to lbfgs_evaluate 
	 * @return gamma
	 */
	float64_t get_gamma(void* instance);

	/** call-back evaluate method for lbfgs
	 *
	 * @param obj object parameters required for loss calculation
	 * @param parameters current state of variables of target function
	 * @param gradient stores gradient computed by this method
	 * @param dim dimensions
	 * @param step step in linesearch
	 */
	static float64_t lbfgs_evaluate(void *obj, const float64_t *parameters, float64_t *gradient, const int dim, const float64_t step);

	/** initialize */
	void init();

protected:
	/** machine to be used for  GBoosting */
	CMachine* m_machine;

	/** loss function */
	CLossFunction* m_loss;

	/** num of iterations */
	int32_t m_num_iter;

	/** subset fraction */
	float64_t m_subset_frac;

	/** learning_rate */
	float64_t m_learning_rate;

	/** array of weak learners */
	CDynamicObjectArray* m_weak_learners;

	/** gamma - weak learner weights */
	CDynamicArray<float64_t>* m_gamma;
};
}/* shogun */

#endif /* _StochasticGBMachine_H__ */
