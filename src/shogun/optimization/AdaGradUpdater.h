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

#ifndef ADAGRADUPDATER_H
#define ADAGRADUPDATER_H
#include <shogun/optimization/DescendUpdaterWithCorrection.h>
#include <shogun/optimization/LearningRate.h>
namespace shogun
{
/** @brief The class implements the AdaGrad method.
 *
 * 	\f[
 *	\begin{array}{l}
 *	g_\theta={(\frac{ \partial f(\cdot) }{\partial \theta })}^2+g_\theta\\
 *	d_\theta=\alpha\frac{1}{\sqrt{g_\theta+\epsilon}}\frac{ \partial f(\cdot) }{\partial \theta }\\
 *	\end{array}
 *	\f]
 *
 * where
 * \f$ \frac{ \partial f(\cdot) }{\partial \theta } \f$ is a negative descend direction (eg, gradient) wrt \f$\theta\f$,
 * \f$\epsilon\f$ is used to avoid dividing by 0,
 * \f$ \alpha \f$ is a build-in learning rate
 * \f$d_\theta\f$ is a corrected negative descend direction.
 *
 * Duchi, John, Elad Hazan, and Yoram Singer. "Adaptive subgradient methods for online learning and stochastic optimization."
 * The Journal of Machine Learning Research 12 (2011): 2121-2159.
 */
 class AdaGradUpdater: public DescendUpdaterWithCorrection
{
public:
	/* Constructor */
	AdaGradUpdater();

	/** Parameterized Constructor
	 * 
	 * @param learning_rate learning_rate
	 * @param epsilon epsilon
	 */
	AdaGradUpdater(float64_t learning_rate,float64_t epsilon);

	/* Destructor */
	virtual ~AdaGradUpdater();

	/** Set learning rate
	 *
	 * @param learning_rate learning rate
	 */
	virtual void set_learning_rate(float64_t learning_rate);

	/** Set epsilon
	 *
	 * @param epsilon epsilon must be positive
	 */
	virtual void set_epsilon(float64_t epsilon);

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
	 *
	 */
	virtual void update_variable(SGVector<float64_t> variable_reference,
		SGVector<float64_t> raw_negative_descend_direction,
		float64_t learning_rate);

protected:
	/** Get the negative descend direction given current variable  and gradient 
	 *
	 * It will be called at update_variable()
	 *
	 * @param variable current variable (eg, \f$\theta\f$)
	 * @param gradient current gradient (eg, \f$ \frac{ \partial f(\cdot) }{\partial \theta }\f$)
	 * @param idx the index of the variable
	 * @param learning_rate learning rate (for AdaGrad, learning_rate is NOT used because there is a build-in
	 * learning_rate)
	 * 
	 * @return negative descend direction (that is, \f$d_\theta\f$)
	 */
	virtual float64_t get_negative_descend_direction(float64_t variable,
		float64_t gradient, index_t idx, float64_t learning_rate);

	/** learning_rate \f$ \alpha \f$ at iteration */
	float64_t m_build_in_learning_rate;

	/** \f$ epsilon \f$ */
	float64_t m_epsilon;

	/** \f$ g_\theta \f$ */
	SGVector<float64_t> m_gradient_accuracy;
private:
	/**  Init */
	void init();
};

}
#endif
