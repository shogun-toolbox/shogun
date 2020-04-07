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

#ifndef RMSPROPUPDATER_H
#define RMSPROPUPDATER_H
#include <shogun/optimization/DescendUpdaterWithCorrection.h>
namespace shogun
{
/** @brief The class implements the RmsProp method.
 *
 *  \f[
 *	\begin{array}{l}
 *	g_\theta=(1-\lambda){(\frac{ \partial f(\cdot) }{\partial \theta })}^2+\lambda g_\theta\\
 *	d_\theta=\alpha\frac{1}{\sqrt{g_\theta+\epsilon}}\frac{ \partial f(\cdot) }{\partial \theta }\\
 *	\end{array}
 *	\f]
 *
 * where
 * \f$ \frac{ \partial f(\cdot) }{\partial \theta } \f$ is a negative descend direction (eg, gradient) wrt \f$\theta\f$,
 * \f$\lambda\f$ is a decay factor,
 * \f$\epsilon\f$ is used to avoid dividing by 0,
 * \f$ \alpha \f$ is a build-in learning rate
 * \f$d_\theta\f$ is a corrected negative descend direction.
 *
 * where \f$\lambda\f$ is a learning rate.
 */
class RmsPropUpdater: public DescendUpdaterWithCorrection
{
public:
	/* Constructor */
	RmsPropUpdater();

	/** Parameterized Constructor
	 *
	 * @param learning_rate learning_rate
	 * @param epsilon epsilon 
	 * @param decay_factor decay_factor
	 */
	RmsPropUpdater(float64_t learning_rate,float64_t epsilon,float64_t decay_factor);

	/* Destructor */
	~RmsPropUpdater() override;

	/** returns the name of the class
	 *
	 * @return name RmsPropUpdate
	 */
	const char* get_name() const override { return "RmsPropUpdater"; }


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


	/** Update the target variable based on the given negative descend direction
	 *
	 * Note that this method will update the target variable in place.
	 * This method will be called by FirstOrderMinimizer::minimize()
	 * 
	 * @param variable_reference a reference of the target variable
	 * @param raw_negative_descend_direction the negative descend direction given the current value
	 * @param learning_rate learning rate
	 */
	void update_variable(SGVector<float64_t> variable_reference,
		SGVector<float64_t> raw_negative_descend_direction, float64_t learning_rate) override;
protected:
	/** Get the negative descend direction given current variable  and gradient 
	 *
	 * It will be called at update_variable()
	 *
	 * @param variable current variable (eg, \f$\theta\f$)
	 * @param gradient current gradient (eg, \f$ \frac{ \partial f(\cdot) }{\partial \theta }\f$)
	 * @param idx the index of the variable
	 * @param learning_rate learning rate (for RmsProp, learning_rate is NOT used because there is a build-in learning_rate)
	 * 
	 * @return negative descend direction (that is, \f$d_\theta\f$)
	 */
	float64_t get_negative_descend_direction(float64_t variable,
		float64_t gradient, index_t idx, float64_t learning_rate) override;

	/** learning_rate \f$\alpha\f$ at iteration */
	float64_t m_build_in_learning_rate;

	/** \f$ \epsilon \f$ */
	float64_t m_epsilon;

	/** decay term \f$ \lambda \f$ */
	float64_t m_decay_factor;

	/** \f$ g_\theta \f$ */
	SGVector<float64_t> m_gradient_accuracy;
private:
	/**  Init */
	void init();
};

}
#endif
