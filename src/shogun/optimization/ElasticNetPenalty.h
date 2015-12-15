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

#ifndef ELASTICNETPENALTY_H
#define ELASTICNETPENALTY_H
#include <shogun/optimization/SparsePenalty.h>
#include <shogun/optimization/L1Penalty.h>
#include <shogun/optimization/L2Penalty.h>
#include <shogun/lib/config.h>
#include <shogun/mathematics/Math.h>
namespace shogun
{
/** @brief The is the base class for ElasticNet penalty/regularization within the FirstOrderMinimizer framework.
 *
 * For ElasticNet penalty, \f$ElasticNet(w)\f$
 * \f[
 * ElasticNet(w)= \lambda \| w \|_1 + (1.0-\lambda) \| w \|_2
 * \f]
 * where \f$\lambda\f$ is the l1_ratio.
 */

class ElasticNetPenalty: public SparsePenalty
{
public:
	ElasticNetPenalty()
		:SparsePenalty() {init();}

	virtual ~ElasticNetPenalty()
	{
		delete m_l1_penalty;
		delete m_l2_penalty;
	}

	virtual void set_l1_ratio(float64_t ratio)
	{
		REQUIRE(ratio>0.0 && ratio<1.0, "");
		m_l1_ratio=ratio;
	}

	/** Given the value of a target variable,
	 * this method returns the penalty of the variable 
	 *
	 * @param variable value of the variable
	 * @return penalty of the variable
	 */
	virtual float64_t get_penalty(float64_t variable)
	{
		check_ratio();
		float64_t penalty=m_l1_ratio*m_l1_penalty->get_penalty(variable);
		penalty+=(1.0-m_l1_ratio)*m_l2_penalty->get_penalty(variable);
		return penalty;
	}

	virtual float64_t get_penalty_gradient(float64_t variable,
		float64_t gradient_of_variable)
	{
		check_ratio();
		float64_t grad=m_l1_ratio*m_l1_penalty->get_penalty_gradient(variable, gradient_of_variable);
		grad+=(1.0-m_l1_ratio)*m_l2_penalty->get_penalty_gradient(variable, gradient_of_variable);
		return grad;
	}

	virtual void set_rounding_eplison(float64_t eplison)
	{
		m_l1_penalty->set_rounding_eplison(eplison);
	}

	virtual void update_variable_for_proximity(SGVector<float64_t> variable,
		float64_t proximal_weight)
	{
		check_ratio();
		m_l1_penalty->update_variable_for_proximity(variable, proximal_weight*m_l1_ratio);
	}

	/** Update a context object to store mutable variables
	 * used in learning rate
	 *
	 * @param context a context object
	 */
	virtual void update_context(CMinimizerContext* context)
	{
		REQUIRE(context, "Context must set\n");
		m_l1_penalty->update_context(context);
		m_l2_penalty->update_context(context);
	}

	/** Load the given context object to restore mutable variables
	 *
	 * @param context a context object
	 */
	virtual void load_from_context(CMinimizerContext* context)
	{
		REQUIRE(context, "Context must set\n");
		m_l1_penalty->load_from_context(context);
		m_l2_penalty->load_from_context(context);
	}

	virtual float64_t get_sparse_variable(float64_t variable, float64_t penalty_weight)
	{
		check_ratio();
		return m_l1_penalty->get_sparse_variable(variable, penalty_weight*m_l1_ratio);
	}
protected:
	virtual void check_ratio()
	{
		REQUIRE(m_l1_ratio>0, "l1_ratio must set\n");
	}

	float64_t m_l1_ratio;
	L1Penalty* m_l1_penalty;
	L2Penalty* m_l2_penalty;

private:
	void init()
	{
		m_l1_ratio=0;
		m_l1_penalty=new L1Penalty();
		m_l2_penalty=new L2Penalty();
	}
};

}

#endif
