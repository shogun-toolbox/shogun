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
#include <shogun/optimization/SVRGMinimizer.h>
#include <shogun/optimization/SGDMinimizer.h>
#include <shogun/optimization/GradientDescendUpdater.h>
#include <shogun/optimization/SparsePenalty.h>
using namespace shogun;

SVRGMinimizer::SVRGMinimizer()
	:FirstOrderStochasticMinimizer()
{
	init();
}

SVRGMinimizer::~SVRGMinimizer()
{
}

SVRGMinimizer::SVRGMinimizer(FirstOrderSAGCostFunction *fun)
	:FirstOrderStochasticMinimizer(fun)
{
	init();
}

void SVRGMinimizer::init()
{
	m_num_sgd_passes=0;
	m_svrg_interval=0;
	m_average_gradient=SGVector<float64_t>();
	m_previous_variable=SGVector<float64_t>();
}

void SVRGMinimizer::init_minimization()
{
	FirstOrderStochasticMinimizer::init_minimization();
	REQUIRE(m_num_sgd_passes>=0, "sgd_passes must set\n");
	REQUIRE(m_svrg_interval>0, "svrg_interval must set\n");
	FirstOrderSAGCostFunction *fun=dynamic_cast<FirstOrderSAGCostFunction *>(m_fun);
	REQUIRE(fun,"the cost function must be a stochastic average gradient cost function\n");
	if (m_num_sgd_passes>0)
	{
		SGDMinimizer sgd(fun);
		sgd.set_number_passes(m_num_sgd_passes);
		sgd.set_gradient_updater(m_gradient_updater);
		sgd.set_penalty_weight(m_penalty_weight);
		sgd.set_penalty_type(m_penalty_type);
		sgd.set_learning_rate(m_learning_rate);
		sgd.minimize();
		m_iter_counter+=sgd.get_iteration_counter();
	}
}

float64_t SVRGMinimizer::minimize()
{
	init_minimization();

	SGVector<float64_t> variable_reference=m_fun->obtain_variable_reference();
	FirstOrderSAGCostFunction *fun=dynamic_cast<FirstOrderSAGCostFunction *>(m_fun);
	REQUIRE(fun,"the cost function must be a stochastic average gradient cost function\n");
	for(;m_cur_passes<(m_num_passes-m_num_sgd_passes);m_cur_passes++)
	{
		if(m_cur_passes%m_svrg_interval==0)
		{
			if(m_previous_variable.vlen==0)
				m_previous_variable=SGVector<float64_t>(variable_reference.vlen);

			std::copy(variable_reference.vector, variable_reference.vector+variable_reference.vlen, m_previous_variable.vector);
			m_average_gradient=fun->get_average_gradient();
		}
		fun->begin_sample();
		while(fun->next_sample())
		{
			m_iter_counter++;
			float64_t learning_rate=1.0;
			if(m_learning_rate)
				learning_rate=m_learning_rate->get_learning_rate(m_iter_counter);

			SGVector<float64_t> grad_new=m_fun->get_gradient();
			SGVector<float64_t> var(variable_reference.vlen);
			std::copy(variable_reference.vector, variable_reference.vector+variable_reference.vlen, var.vector);

			std::copy(m_previous_variable.vector, m_previous_variable.vector+m_previous_variable.vlen, variable_reference.vector);
			SGVector<float64_t> grad_old=m_fun->get_gradient();

			std::copy(var.vector, var.vector+var.vlen, variable_reference.vector);
			for(index_t idx=0; idx<grad_new.vlen; idx++)
				grad_new[idx]+=(m_average_gradient[idx]-grad_old[idx]);

			update_gradient(grad_new,variable_reference);
			m_gradient_updater->update_variable(variable_reference,grad_new,learning_rate);

			do_proximal_operation(variable_reference);
		}
	}
	float64_t cost=m_fun->get_cost();
	return cost+get_penalty(variable_reference);
}
