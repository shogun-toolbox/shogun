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

#ifndef L1PENALTY_H
#define L1PENALTY_H
#include <shogun/optimization/SparsePenalty.h>
#include <shogun/lib/config.h>
#include <shogun/mathematics/Math.h>
namespace shogun
{
/** @brief The is the base class for L1 penalty/regularization within the FirstOrderMinimizer framework.
 *
 * For L1 penalty, \f$L1(w)\f$
 * \f[
 * L1(w)=\| w \|_1 = \sum_i \| w_i \|
 * \f]
 *
 * This class implements the soft-threshold method.
 * Reference:
 * Proximal gradient method
 * www.eecs.berkeley.edu/~elghaoui/Teaching/EE227A/lecture18.pdf
 */

class L1Penalty: public SparsePenalty
{
public:
	L1Penalty()
		:SparsePenalty() {init();}

	virtual ~L1Penalty() {}

	/** Given the value of a target variable,
	 * this method returns the penalty of the variable 
	 *
	 * @param variable value of the variable
	 * @return penalty of the variable
	 */
	virtual float64_t get_penalty(float64_t variable) {return CMath::abs(variable);}


	/** Return the gradient of the penalty wrt a target variable
	 *
	 * Note that for L1 penalty we do not compute the gradient/sub-gradient in our implementation.
	 * Instead, we do a proximal projection.
	 *
	 * @param variable value of a target variable
	 * @param gradient_of_variable unregularized/unpenalized gradient of the variable
	 * @return the gradient of the penalty wrt the variable
	 */

	virtual float64_t get_penalty_gradient(float64_t variable,
		float64_t gradient_of_variable) {return 0.0;}

	/** Set the rounding epsilon
	 *
	 * @param epsilon rounding epsilon
	 *
	 */
	virtual void set_rounding_epsilon(float64_t epsilon)
	{
		REQUIRE(epsilon>=0,"Rounding epsilon (%f) should be non-negative\n", epsilon);
		m_rounding_epsilon=epsilon;
	}

	/** Do proximal projection/operation in place
	 * @param variable the raw variable
	 * @param proximal_weight weight of the penalty
	 */
	virtual void update_variable_for_proximity(SGVector<float64_t> variable,
		float64_t proximal_weight)
	{
		for(index_t idx=0; idx<variable.vlen; idx++)
			variable[idx]=get_sparse_variable(variable[idx], proximal_weight);
	}

	/** Update a context object to store mutable variables
	 * used in learning rate
	 *
	 * @param context a context object
	 */
	virtual void update_context(CMinimizerContext* context)
	{
		REQUIRE(context, "Context must set\n");
	}

	/** Load the given context object to restore mutable variables
	 *
	 * @param context a context object
	 */
	virtual void load_from_context(CMinimizerContext* context)
	{
		REQUIRE(context, "Context must set\n");
	}

	/** Get the sparse variable
	 * @param variable the raw variable
	 * @param penalty_weight weight of the penalty
	 * @return sparse value of the variable
	 */
	virtual float64_t get_sparse_variable(float64_t variable, float64_t penalty_weight)
	{
	  if (variable>0.0)
	  {
		  variable-=penalty_weight;
		  if (variable<0.0)
			  variable=0.0;
	  }
	  else
	  {
		  variable+=penalty_weight;
		  if (variable>0.0)
			  variable=0.0;
	  }
	  if (CMath::abs(variable)<m_rounding_epsilon)
		  variable=0.0;
	  return variable;
	}
protected:
	/** rounding epsilon */
	float64_t m_rounding_epsilon;

private:
	/** init */
	void init()
	{
		m_rounding_epsilon=1e-8;
	}
};

}

#endif
