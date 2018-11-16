 /* * Copyright (c) The Shogun Machine Learning Toolbox
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
#include <shogun/optimization/SMDMinimizer.h>
#include <shogun/base/Parameter.h>
using namespace shogun;

SMDMinimizer::SMDMinimizer()
	:FirstOrderStochasticMinimizer()
{
	init();
}

SMDMinimizer::~SMDMinimizer()
{
	SG_UNREF(m_mapping_fun);
}

SMDMinimizer::SMDMinimizer(FirstOrderStochasticCostFunction *fun)
	:FirstOrderStochasticMinimizer(fun)
{
	init();
}

float64_t SMDMinimizer::minimize()
{
	REQUIRE(m_mapping_fun, "Mapping function must set\n");
	init_minimization();
	SGVector<float64_t> variable_reference=m_fun->obtain_variable_reference();
	SGVector<float64_t> dual_variable=m_mapping_fun->get_dual_variable(variable_reference);
	FirstOrderStochasticCostFunction *fun=dynamic_cast<FirstOrderStochasticCostFunction *>(m_fun);
	REQUIRE(fun,"the cost function must be a stochastic cost function\n");
	for(;m_cur_passes<m_num_passes;m_cur_passes++)
	{
		fun->begin_sample();
		while(fun->next_sample())
		{
			m_iter_counter++;
			float64_t learning_rate=1.0;
			if(m_learning_rate)
				learning_rate=m_learning_rate->get_learning_rate(m_iter_counter);

			SGVector<float64_t> grad=m_fun->get_gradient();
			update_gradient(grad,variable_reference);
			m_gradient_updater->update_variable(dual_variable,grad,learning_rate);
			m_mapping_fun->update_variable(variable_reference,dual_variable);
		}
	}
	float64_t cost=m_fun->get_cost();
	return cost+get_penalty(variable_reference);
}

void SMDMinimizer::init()
{
	m_mapping_fun=NULL;
	SG_ADD((CSGObject **)&m_mapping_fun, "SMDMinimizer__m_mapping_fun",
		"m_mapping_fun in SMDMinimizer");
}

void SMDMinimizer::set_mapping_function(MappingFunction* mapping_fun)
{
	REQUIRE(mapping_fun, "mapping/projection function must be set\n");
	if(m_mapping_fun!=mapping_fun)
	{
		SG_REF(mapping_fun);
		SG_UNREF(m_mapping_fun);
		m_mapping_fun=mapping_fun;
	}
}

void SMDMinimizer::init_minimization()
{
	FirstOrderStochasticMinimizer::init_minimization();
}
