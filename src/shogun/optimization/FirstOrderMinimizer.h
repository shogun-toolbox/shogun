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

#ifndef FIRSTORDERMINIMIZER_H
#define FIRSTORDERMINIMIZER_H
#include <shogun/optimization/FirstOrderCostFunction.h>
#include <shogun/optimization/Minimizer.h>
#include <shogun/optimization/Penalty.h>
namespace shogun
{

/** @brief The first order minimizer base class.
 *
 * This class gives the interface of a first-order gradient-based unconstrained minimizer
 *
 * This kind of minimizers will find optimal target variables based on gradient information wrt target variables.
 * For example, the gradient descend method is a minimizer.
 *
 * A minimizer requires the following objects as input:
 * a supported cost function object (eg, FirstOrderCostFunction )
 * a penalty object if regularization is enabled (eg, Penalty )
 *
 */
class FirstOrderMinimizer: public Minimizer
{
public: 
	/** Default constructor */
	FirstOrderMinimizer():Minimizer()
	{
		init();
	}

	/** Constructor
	 * @param fun cost function (user have to manully delete the pointer)
	 */
	FirstOrderMinimizer(FirstOrderCostFunction *fun)
	{
		init();
		set_cost_function(fun);
	}

	/** returns the name of the class
	 *
	 * @return name FirstOrderMinimizer
	 */
	virtual const char* get_name() const { return "FirstOrderMinimizer"; }

	/** Destructor */
	virtual ~FirstOrderMinimizer();

	/** Does minimizer support batch update?
	 * 
	 * @return whether minimizer supports batch update
	 */
	virtual bool supports_batch_update() const=0;

	/** Set cost function used in the minimizer
	 *
	 * @param fun the cost function
	 */
	virtual void set_cost_function(FirstOrderCostFunction *fun);

	/** Unset cost function used in the minimizer
	 *
	 */
	virtual void unset_cost_function(bool is_unref=true)
	{
		if(is_unref)
		{
			SG_UNREF(m_fun);
		}
		m_fun=NULL;
	}

	/** Set the weight of penalty
	 *
	 * @param penalty_weight the weight of penalty, which is positive
	 */
	virtual void set_penalty_weight(float64_t penalty_weight);

	/** Set the type of penalty
	 * For example, L2 penalty
	 *
	 * @param penalty_type the type of penalty. If NULL is given, regularization is not enabled.
	 */
	virtual void set_penalty_type(Penalty* penalty_type);

protected:
	/** Get the penalty given target variables
	 * For L2 penalty,
	 * the target variable is \f$w\f$
	 * and
	 * the value of penalty is \f$\lambda \frac{w^t w}{2}\f$,
	 * where \f$\lambda\f$ is the weight of penalty
	 *
	 *
	 * @param var the variable used in regularization
	 */
	virtual float64_t get_penalty(SGVector<float64_t> var);

	/** Add gradient of the penalty wrt target variables to unpenalized gradient
	 * For least sqaure with L2 penalty,
	 * \f[
	 * L2f(w)=f(w) + L2(w) \f]
	 * where \f$ f(w)=\sum_i{(y_i-w^T x_i)^2}\f$ is the least sqaure cost function
	 * and \f$L2(w)=\lambda \frac{w^t w}{2}\f$ is the L2 penalty
	 *
	 * Target variables is \f$w\f$
	 * Unpenalized gradient is \f$\frac{\partial f(w) }{\partial w}\f$
	 * Gradient of the penalty wrt target variables is \f$\frac{\partial L2(w) }{\partial w}\f$
	 *
	 * @param gradient unpenalized gradient wrt its target variable
	 * @param var the target variable
	 */
	virtual void update_gradient(SGVector<float64_t> gradient, SGVector<float64_t> var);

	/** Cost function */
	FirstOrderCostFunction *m_fun;

	/** the type of penalty*/
	Penalty* m_penalty_type;

	/** the weight of penalty*/
	float64_t m_penalty_weight;

private:
	/**  init */
	void init();
};

}
#endif
