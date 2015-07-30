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

#include <shogun/optimization/FirstOrderPenalizedStochasticCostFunction.h>
using namespace shogun;

CFirstOrderPenalizedStochasticCostFunction::CFirstOrderPenalizedStochasticCostFunction()
	:CFirstOrderStochasticCostFunction()
{
	init();
}

CFirstOrderPenalizedStochasticCostFunction::CFirstOrderPenalizedStochasticCostFunction(
	CFirstOrderStochasticCostFunction* fun)
	:CFirstOrderStochasticCostFunction()
{
	init();
	set_cost_function(fun);
}

CFirstOrderPenalizedStochasticCostFunction::~CFirstOrderPenalizedStochasticCostFunction()
{
	delete m_helper;
}

void CFirstOrderPenalizedStochasticCostFunction::init()
{
	m_fun=NULL;
	m_helper=new CFirstOrderPenalizedCostFunction();
}

void CFirstOrderPenalizedStochasticCostFunction::set_penalty_weight(float64_t penalty_weight)
{
	m_helper->set_penalty_weight(penalty_weight);
}

void CFirstOrderPenalizedStochasticCostFunction::set_penalty_type(CPenalty* penalty_type)
{
	m_helper->set_penalty_type(penalty_type);
}

void CFirstOrderPenalizedStochasticCostFunction::set_cost_function(
	CFirstOrderStochasticCostFunction* fun)
{
	//REQUIRE(fun,"cost function must not be NULL\n");
	//REQUIRE(this!=fun,"cost function must not be itself")
	if(m_fun!=fun)
	{
		m_fun=fun;
		m_helper->set_cost_function(m_fun);
	}
}

float64_t CFirstOrderPenalizedStochasticCostFunction::get_cost()
{
	return m_helper->get_cost();
}

SGVector<float64_t> CFirstOrderPenalizedStochasticCostFunction::obtain_variable_reference()
{
	return m_helper->obtain_variable_reference();
}

SGVector<float64_t> CFirstOrderPenalizedStochasticCostFunction::get_gradient()
{
	return m_helper->get_gradient();
}

void CFirstOrderPenalizedStochasticCostFunction::begin_sample()
{
	//REQUIRE(m_fun,"cost function must set\n");
	m_fun->begin_sample();
}

bool CFirstOrderPenalizedStochasticCostFunction::next_sample()
{
	//REQUIRE(m_fun,"cost function must set\n");
	return m_fun->next_sample();
}
