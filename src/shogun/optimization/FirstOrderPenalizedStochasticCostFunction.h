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

#ifndef FIRSTORDERPENALIZEDSTOCHASTICCOSTFUNCTION_H
#define FIRSTORDERPENALIZEDSTOCHASTICCOSTFUNCTION_H
#include <shogun/lib/config.h>
#include <shogun/optimization/FirstOrderStochasticCostFunction.h>
#include <shogun/optimization/FirstOrderPenalizedCostFunction.h>
namespace shogun
{
/** @brief The first order penalized stochastic cost function base class.
 *
 * note that the cost function must be the finite sum of sample-specific cost
 * decorator pattern
 *
 */

#define IGNORE_IN_CLASSLIST
IGNORE_IN_CLASSLIST class CFirstOrderPenalizedStochasticCostFunction: public CFirstOrderStochasticCostFunction
{
public:
	/* constructor */
	CFirstOrderPenalizedStochasticCostFunction();

	/** constructor
	 * @param fun the stochastic cost function to be penalized
	 */
	CFirstOrderPenalizedStochasticCostFunction(CFirstOrderStochasticCostFunction* fun);

	/* destructor */
	virtual ~CFirstOrderPenalizedStochasticCostFunction();

	/** get the penalized cost given current variables 
	 *
	 * note that the cost is the finite sum of sample-specific cost
	 * @return cost
	 */
	virtual float64_t get_cost();

	/** obtain reference of objetive variables 
	 *
	 * @return reference of objetive variables
	 */
	virtual SGVector<float64_t> obtain_variable_reference();

	/** get the penalized gradient value of variables for a sample
	 * 
	 * Note that must call begin_sample() to initialize the sample sequence
	 * then must call next_sample() to generate a sample (say, sample-i)
	 * this method returns the penalized gradient value of variables for the sample-i
	 *
	 * @return penalized gradient of variables for the sample
	 */
	virtual SGVector<float64_t> get_gradient();

	/** set the weight of penalty
	 * for example, for L2 penalty
	 * the penalized cost function is =unpenalized cost function+penalty_weight*L2_penalty
	 *
	 * @param penalty_weight the weight of penalty
	 */
	virtual void set_penalty_weight(float64_t penalty_weight);

	/** set the type of penalty
	 * for example, L2 penalty
	 *
	 * @param penalty_type the type of penalty
	 */
	virtual void set_penalty_type(CPenalty* penalty_type);

	/** set the cost function
	 *
	 * @param fun stochastic cost function to be penalized
	 */
	virtual void set_cost_function(CFirstOrderStochasticCostFunction* fun);

	/* initialize to get samples
	 */
	virtual void begin_sample();

	/* get next sample
	 *
	 * @return false if reach the end of sample sequence
	 * */
	virtual bool next_sample();

protected:
	/*  wrapped stochastic cost function */
	CFirstOrderStochasticCostFunction* m_fun;
	/* reuse the penalized  non-stochastic cost function */
	CFirstOrderPenalizedCostFunction* m_helper;

private:
	/*  init */
	void init();
};

}
#endif
