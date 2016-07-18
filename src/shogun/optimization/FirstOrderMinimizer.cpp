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

#include <shogun/base/Parameter.h>
#include <shogun/optimization/FirstOrderMinimizer.h>
using namespace shogun;

FirstOrderMinimizer::~FirstOrderMinimizer()
{
	SG_UNREF(m_fun);
	SG_UNREF(m_penalty_type);
}

void FirstOrderMinimizer::set_cost_function(FirstOrderCostFunction *fun)
{
	REQUIRE(fun,"The cost function must be not NULL\n");
	if(m_fun != fun)
	{
		SG_REF(fun);
		SG_UNREF(m_fun);
		m_fun=fun;
	}
}

void FirstOrderMinimizer::set_penalty_type(Penalty* penalty_type)
{
	if(m_penalty_type != penalty_type)
	{
		SG_REF(penalty_type);
		SG_UNREF(m_penalty_type);
		m_penalty_type=penalty_type;
	}
}

void FirstOrderMinimizer::set_penalty_weight(float64_t penalty_weight)
{
	REQUIRE(penalty_weight>0,"The weight of penalty must be positive\n");
	m_penalty_weight=penalty_weight;
}

float64_t FirstOrderMinimizer::get_penalty(SGVector<float64_t> var)
{
	float64_t penalty=0.0;
	if(m_penalty_type)
	{
		REQUIRE(m_penalty_weight>0,"The weight of penalty must be set first\n");
		for(index_t idx=0; idx<var.vlen; idx++)
			penalty+=m_penalty_weight*m_penalty_type->get_penalty(var[idx]);
	}
	return penalty;
}

void FirstOrderMinimizer::update_gradient(SGVector<float64_t> gradient, SGVector<float64_t> var)
{
	if(m_penalty_type)
	{
		REQUIRE(m_penalty_weight>0,"The weight of penalty must be set first\n");
		for(index_t idx=0; idx<var.vlen; idx++)
		{
			float64_t grad=gradient[idx];
			float64_t variable=var[idx];
			gradient[idx]+=m_penalty_weight*m_penalty_type->get_penalty_gradient(variable,grad);
		}
	}
}

void FirstOrderMinimizer::init()
{
	m_fun=NULL;
	m_penalty_type=NULL;
	m_penalty_weight=0;
	SG_ADD(&m_penalty_weight, "FirstOrderMinimizer__m_penalty_weight",
		"penalty_weight in FirstOrderMinimizer", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject **)&m_penalty_type, "FirstOrderMinimizer__m_penalty_type",
		"penalty_type in FirstOrderMinimizer", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject **)&m_fun, "FirstOrderMinimizer__m_fun",
		"penalty_fun in FirstOrderMinimizer", MS_NOT_AVAILABLE);
}
