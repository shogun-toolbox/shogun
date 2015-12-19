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
/** @brief The class implements the AdaDelta method.
 *	\f[
 *	\begin{array}{l}
 *	g_\theta=(1-\lambda){(\frac{ \partial f(\cdot) }{\partial \theta })}^2+\lambda g_\theta\\
 *	d_\theta=\alpha\frac{\sqrt{s_\theta+\epsilon}}{\sqrt{g_t+\epsilon}}\frac{ \partial f(\cdot) }{\partial \theta }\\
 *	s_\theta=(1-\lambda){(d_\theta)}^2+\lambda s_\theta
 *	\end{array}
 *	\f]
 *
 * where
 * \f$ \frac{ \partial f(\cdot) }{\partial \theta } \f$ is a negative descend direction (eg, gradient) wrt \f$\theta$\f,
 * \f$\lambda\f$ is a decay factor,
 * \f$\epsilon$\f is used to avoid dividing by 0,
 * \f$ \alpha \f$ is a build-in learning rate
 * \f$d_\theta$\f is a corrected negative descend direction.
 * 
 * Reference:
 * Matthew D. Zeiler, ADADELTA: An Adaptive Learning Rate Method, arXiv:1212.5701
 *
 * */
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
	/** Get the negative descend direction given current variable  and gradient 
	 *
	 * It will be called at update_variable()
	 *
	 * @param variable current variable (eg, \f$\theta$\f)
	 * @param gradient current gradient (eg, \f$ \frac{ \partial f(\cdot) }{\partial \theta }$\f)
	 * @param idx the index of the variable
	 * @param learning_rate learning rate (for AdaDelta, learning_rate is NOT used because there is a build-in
	 * learning_rate)
	 * 
	 * @return negative descend direction (that is, \f$\d_\theta$\f)
	 */
	virtual float64_t get_negative_descend_direction(float64_t variable,
		float64_t gradient, index_t idx, float64_t learning_rate);

	/** learning_rate \f$\alpha$\f at iteration */
	float64_t m_build_in_learning_rate;

	/** \f$epsilon$\f */
	float64_t m_epsilon;

	/** decay term \f$\lambda$\f */
	float64_t m_decay_factor;

	/** \f$g_\theta$\f */
	SGVector<float64_t> m_gradient_accuracy;

	/** \f$s_\theta$\f */
	SGVector<float64_t> m_gradient_delta_accuracy;
private:
	/**  Init */
	void init();
};

}
#endif
