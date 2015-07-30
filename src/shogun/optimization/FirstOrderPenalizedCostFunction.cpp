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

#include <shogun/optimization/FirstOrderPenalizedCostFunction.h>

using namespace shogun;
CFirstOrderPenalizedCostFunction::CFirstOrderPenalizedCostFunction()
	: CFirstOrderCostFunction()
{
	init();
}

CFirstOrderPenalizedCostFunction::CFirstOrderPenalizedCostFunction(CFirstOrderCostFunction* fun)
	: CFirstOrderCostFunction()
{
	init();
	set_cost_function(fun);
}

CFirstOrderPenalizedCostFunction::~CFirstOrderPenalizedCostFunction()
{
}

void CFirstOrderPenalizedCostFunction::set_cost_function(CFirstOrderCostFunction* fun)
{
	//REQUIRE(fun,"cost function must not be NULL\n");
	//REQUIRE(this!=fun,"cost function must not be itself\n");
	if(m_fun!=fun)
	{
		m_fun=fun;
		m_variable_reference=m_fun->obtain_variable_reference();
	}
}

void CFirstOrderPenalizedCostFunction::set_penalty_weight(float64_t penalty_weight)
{
	//REQUIRE(penalty_weight>0,"penalty_weight must be positive\n");
	m_penalty_weight=penalty_weight;
}

void CFirstOrderPenalizedCostFunction::set_penalty_type(CPenalty* penalty_type)
{
	//REQUIRE(penalty_type,"the type of penalty_type must not be NULL\n");
	if(m_penalty_type!=penalty_type)
	{
		m_penalty_type=penalty_type;
	}
}

void CFirstOrderPenalizedCostFunction::init()
{
	m_fun=NULL;
	m_penalty_type=NULL;
	m_penalty_weight=0;
	m_variable_reference=SGVector<float64_t>();
}

float64_t CFirstOrderPenalizedCostFunction::get_cost()
{

	//REQUIRE(m_penalty_weight>0,"penalty_weight must be positive\n");
	//REQUIRE(m_fun,"cost_function must set\n");
	//REQUIRE(m_penalty_type,"the type of penalty must set\n");
	//REQUIRE(m_variable_reference.vlen>0,"the reference of objective variable must set\n");
	float64_t cost=m_fun->get_cost();

	for(auto idx=0; idx<m_variable_reference.vlen; idx++)
		cost+=m_penalty_weight*m_penalty_type->get_penalty(m_variable_reference[idx]);

	return cost;
}

SGVector<float64_t> CFirstOrderPenalizedCostFunction::obtain_variable_reference()
{
	//REQUIRE(m_fun,"cost function must set\n");
	return m_fun->obtain_variable_reference();
}

SGVector<float64_t> CFirstOrderPenalizedCostFunction::get_gradient()
{
	//REQUIRE(m_penalty_weight>0,"penalty_weight must be positive\n");
	//REQUIRE(m_fun,"cost function must set\n");
	//REQUIRE(m_penalty_type,"the type of penalty must set\n");
	SGVector<float64_t> grad=m_fun->get_gradient();
	//REQUIRE(m_variable_reference.vlen>0,"the reference of objective variable must set\n");
	//REQUIRE(m_variable_reference.vlen==grad.vlen,
		//"number (%d) of objective variables must match number (%d) of their gradients\n",
		//m_variable_reference.vlen,grad.vlen);

	for(auto idx=0; idx<m_variable_reference.vlen; idx++)
	{
		float64_t gradient=grad[idx];
		float64_t variable=m_variable_reference[idx];
		grad[idx]+=m_penalty_weight*m_penalty_type->get_gradient_wrt_penalty(variable,gradient);
	}

	return grad;
}
