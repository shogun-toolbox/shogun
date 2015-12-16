/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2015 Wu Lin
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
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
 *
 */

#ifndef ADADELTAUPDATER_H
#define ADADELTAUPDATER_H
#include <shogun/optimization/DescendUpdaterWithCorrection.h>
#include <shogun/optimization/LearningRate.h>
namespace shogun
{
/** @brief The class implements the AdaDelta method. */
class AdaDeltaUpdater: public DescendUpdaterWithCorrection
{
public:
	/* Constructor */
	AdaDeltaUpdater();

	/* Destructor */
	virtual ~AdaDeltaUpdater();

	/** Set learning rate
	 *
	 * @param learning_rate learning rate
	 */
	virtual void set_learning_rate(float64_t learning_rate);

	/** Set epsilon
	 *
	 * @param epsilon epsilon
	 */
	virtual void set_epsilon(float64_t epsilon);

	/** Set decay_factor
	 *
	 * @param decay_factor decay factor
	 */
	virtual void set_decay_factor(float64_t decay_factor);

	/** Update a context object to store mutable variables
	 *
	 * This method will be called by
	 * FirstOrderMinimizer::save_to_context()
	 *
	 * @param context a context object
	 */
	virtual void update_context(CMinimizerContext* context);

	/** Return a context object which stores mutable variables
	 * Usually it is used in serialization.
	 *
	 * This method will be called by
	 * FirstOrderMinimizer::load_from_context(CMinimizerContext* context)
	 *
	 * @return a context object
	 */
	virtual void load_from_context(CMinimizerContext* context);

	/** Update the target variable based on the given negative descend direction
	 *
	 * Note that this method will update the target variable in place.
	 * This method will be called by FirstOrderMinimizer::minimize()
	 * 
	 * @param variable_reference a reference of the target variable
	 * @param raw_negative_descend_direction the negative descend direction given the current value
	 * @param learning_rate learning rate
	 */
	virtual void update_variable(SGVector<float64_t> variable_reference,
		SGVector<float64_t> raw_negative_descend_direction, float64_t learning_rate);

protected:
	/** Get the negative descend direction given current variable and gradient
	 *
	 * It will be called at update_variable()
	 *
	 * @param variable current variable
	 * @param gradient current gradient
	 * @param idx the index of the variable
	 * @param learning_rate learning rate
	 * 
	 * @return negative descend direction (that is, the given gradient in the class)
	 */
	virtual float64_t get_negative_descend_direction(float64_t variable,
		float64_t gradient, index_t idx, float64_t learning_rate);

	/** learning_rate at iteration */
	float64_t m_build_in_learning_rate;

	/** epsilon */
	float64_t m_epsilon;

	/** decay term */
	float64_t m_decay_factor;

	SGVector<float64_t> m_gradient_accuracy;
	SGVector<float64_t> m_gradient_delta_accuracy;
private:
	/**  Init */
	void init();
};

}
#endif
