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
#include <shogun/optimization/SMIDASMinimizer.h>
#include <shogun/lib/config.h>
#include <shogun/optimization/L1Penalty.h>
#include <shogun/optimization/GradientDescendUpdater.h>

#include <utility>

using namespace shogun;

SMIDASMinimizer::SMIDASMinimizer()
	:SMDMinimizer()
{
	init();
}

SMIDASMinimizer::~SMIDASMinimizer()
{
}

SMIDASMinimizer::SMIDASMinimizer(std::shared_ptr<FirstOrderStochasticCostFunction >fun)
	:SMDMinimizer(std::move(fun))
{
	init();
}

float64_t SMIDASMinimizer::minimize()
{
	require(m_mapping_fun, "Mapping function must set");
	init_minimization();
	SGVector<float64_t> variable_reference=m_fun->obtain_variable_reference();

	if(m_dual_variable.vlen==0)
		m_dual_variable=m_mapping_fun->get_dual_variable(variable_reference);
	else
	{
		require(m_dual_variable.vlen==variable_reference.vlen,
			"The length ({}) of dual variable must match the length ({}) of variable",
			m_dual_variable.vlen, variable_reference.vlen);
	}
	auto penalty_type=std::dynamic_pointer_cast<L1Penalty>(m_penalty_type);
	require(penalty_type,"For now only L1Penalty is supported. Please use the penalty for this minimizer");

	auto fun=m_fun->as<FirstOrderStochasticCostFunction>();
	require(fun,"the cost function must be a stochastic cost function");
	for(;m_cur_passes<m_num_passes;m_cur_passes++)
	{
		fun->begin_sample();
		while(fun->next_sample())
		{
			m_iter_counter++;
			float64_t learning_rate=m_learning_rate->get_learning_rate(m_iter_counter);

			SGVector<float64_t> grad=m_fun->get_gradient();
			m_gradient_updater->update_variable(m_dual_variable,grad, learning_rate);
			penalty_type->update_variable_for_proximity(m_dual_variable, m_penalty_weight*learning_rate);
			m_mapping_fun->update_variable(variable_reference, m_dual_variable);
		}
	}
	float64_t cost=m_fun->get_cost();
	return cost+get_penalty(variable_reference);
}

void SMIDASMinimizer::init()
{
	m_dual_variable=SGVector<float64_t>();
	SG_ADD(&m_dual_variable, "SMIDASMinimizer__m_dual_variable",
		"dual_variable in SMIDASMinimizer");
}

void SMIDASMinimizer::init_minimization()
{
	SMDMinimizer::init_minimization();
	auto updater=
		std::dynamic_pointer_cast<DescendUpdaterWithCorrection>(m_gradient_updater);

	if(updater)
	{
		if (updater->enables_descend_correction())
		{
			io::warn("There is not theoretical guarantee when Descend Correction is enabled");
		}
		auto gradient_updater=
			std::dynamic_pointer_cast<GradientDescendUpdater>(m_gradient_updater);
		if(!gradient_updater)
		{
			io::warn("There is not theoretical guarantee when this updater is used");
		}
	}
	else
	{
		io::warn("There is not theoretical guarantee when this updater is used");
	}
	require(m_learning_rate,"Learning Rate instance must set");
}
