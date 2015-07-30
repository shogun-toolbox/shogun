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

#ifndef FIRSTORDERPENALIZEDCOSTFUNCTION_H
#define FIRSTORDERPENALIZEDCOSTFUNCTION_H
#include <shogun/optimization/FirstOrderCostFunction.h>
#include <shogun/optimization/Penalty.h>

namespace shogun
{
/** @brief This is a first order penalized cost function base class.
 *
 *  decorator pattern
 *
 */

#define IGNORE_IN_CLASSLIST
IGNORE_IN_CLASSLIST class CFirstOrderPenalizedCostFunction: public CFirstOrderCostFunction
{
public:
	/* constructor */
	CFirstOrderPenalizedCostFunction();

	/** constructor
	 *
	 * @param fun cost function to be penalized
	 */
	CFirstOrderPenalizedCostFunction(CFirstOrderCostFunction* fun);

	/* destructor */
	virtual ~CFirstOrderPenalizedCostFunction();

	/** get the penalized cost given current variables 
	 *
	 * @return cost
	 */
	virtual float64_t get_cost();

	/** obtain reference of objetive variables 
	 *
	 * @return reference of objetive variables
	 */
	virtual SGVector<float64_t> obtain_variable_reference();

	/** get the penalized gradient value of variables 
	 *
	 * @return penalized gradient of variables
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
	 * @param fun cost function to be penalized
	 */
	virtual void set_cost_function(CFirstOrderCostFunction* fun);

protected:
	/* cost function to be penalized */
	CFirstOrderCostFunction* m_fun;
	/* the type of penalty*/
	CPenalty* m_penalty_type;
	/* the weight of penalty*/
	float64_t m_penalty_weight;
	/* the reference of the objetive variable*/
	SGVector<float64_t> m_variable_reference;

private:
	/* init */
	void init();
};

}

#endif
