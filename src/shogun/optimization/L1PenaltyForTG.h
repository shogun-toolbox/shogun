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

#ifndef L1PENALTYFORTG_H
#define L1PENALTYFORTG_H
#include <shogun/optimization/L1Penalty.h>
#include <shogun/lib/config.h>

#include <iostream>
namespace shogun
{
/** @brief The is the base class for L1 penalty/regularization within the FirstOrderMinimizer framework.
 *
 * For L1 penalty, \f$L1(w)\f$
 * \f[
 * L1(w)=\| w \|_1 = \sum_i \| w_i \|
 * \f]
 *
 * This class implements the truncated gradient method.
 *
 * Reference:
 * Stochastic Gradient Descent Training for L1-regularized Log-linear Models with Cumulative Penalty
 */

class L1PenaltyForTG: public L1Penalty
{
public:
	/* Constructor */
	L1PenaltyForTG():L1Penalty() { init(); }

	/* Destructor */
	virtual ~L1PenaltyForTG() {}

	/** Do proximal projection/operation in place
	 * @param variable the raw variable
	 * @param penalty_weight weight of the penalty
	 */
	virtual void update_variable_for_proximity(SGVector<float64_t> variable,
		float64_t proximal_weight)
	{
		if(m_q.vlen==0)
		{
			m_q=SGVector<float64_t>(variable.vlen);
			m_q.set_const(0.0);
		}
		else
		{
			REQUIRE(variable.vlen==m_q.vlen,
				"The length of variable (%d) is changed. Last time, the length of variable was %d", variable.vlen, m_q.vlen);
		}
		m_u+=proximal_weight;
		for(index_t idx=0; idx<variable.vlen; idx++)
		{
			float64_t z=variable[idx];
			if(z>0.0)
				variable[idx]=get_sparse_variable(z, m_u+m_q[idx]);
			else if(z<0.0)
				variable[idx]=get_sparse_variable(z, m_u-m_q[idx]);
			m_q[idx]+=variable[idx]-z;
		}
	}

	/** Update a context object to store mutable variables
	 * used in learning rate
	 *
	 * @param context a context object
	 */
	virtual void update_context(CMinimizerContext* context)
	{
		REQUIRE(context, "Context must set\n");
		L1Penalty::update_context(context);
		SGVector<float64_t> value(m_q.vlen);
		std::copy(m_q.vector,
			m_q.vector+m_q.vlen,
			value.vector);
		std::string key="L1PenaltyForTG::m_q";
		context->save_data(key, value);

		key="L1PenaltyForTG::m_u";
		context->save_data(key, m_u);
	}

	/** Load the given context object to restore mutable variables
	 *
	 * @param context a context object
	 */
	virtual void load_from_context(CMinimizerContext* context)
	{
		REQUIRE(context, "Context must set\n");
		L1Penalty::load_from_context(context);
		std::string key="L1PenaltyForTG::m_q";
		SGVector<float64_t> value=context->get_data_sgvector_float64(key);
		m_q=SGVector<float64_t>(value.vlen);
		std::copy(value.vector, value.vector+value.vlen,
			m_q.vector);

		key="L1PenaltyForTG::m_u";
		m_u=context->get_data_float64(key);
	}
protected:
	/** u is defined in Figure 2 of the reference */
	float64_t m_u;
	/** q is defined in Figure 2 of the reference */
	SGVector<float64_t> m_q;

private:
	/** init */
	void init()
	{
		m_u=0;
		m_q=SGVector<float64_t>();
	}

};

}

#endif
