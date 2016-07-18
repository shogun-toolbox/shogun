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
#include <shogun/optimization/lbfgs/LBFGSMinimizer.h>
#include <shogun/optimization/FirstOrderBoundConstraintsCostFunction.h>
#include <shogun/base/Parameter.h>

namespace shogun
{
CLBFGSMinimizer::CLBFGSMinimizer()
	:FirstOrderMinimizer()
{
	init();
}

CLBFGSMinimizer::~CLBFGSMinimizer()
{
}

CLBFGSMinimizer::CLBFGSMinimizer(FirstOrderCostFunction *fun)
	:FirstOrderMinimizer(fun)
{
	FirstOrderBoundConstraintsCostFunction* bound_constraints_fun
		=dynamic_cast<FirstOrderBoundConstraintsCostFunction *>(m_fun);
	if(m_fun && bound_constraints_fun)
	{
		SG_SWARNING("The minimizer does not support constrained minimization. All constraints will be ignored.\n")
	}
	init();
}

void CLBFGSMinimizer::init()
{
	set_lbfgs_parameters();
	m_min_step=1e-6;
	m_xtol=1e-6;
	SG_ADD(&m_linesearch_id, "CLBFGSMinimizer__m_linesearch_id",
		"linesearch_id in CLBFGSMinimizer", MS_NOT_AVAILABLE);
	SG_ADD(&m_m, "CLBFGSMinimizer__m_m",
		"m in CLBFGSMinimizer", MS_NOT_AVAILABLE);
	SG_ADD(&m_max_linesearch, "CLBFGSMinimizer__m_max_linesearch",
		"max_linesearch in CLBFGSMinimizer", MS_NOT_AVAILABLE);
	SG_ADD(&m_max_iterations, "CLBFGSMinimizer__m_max_iterations",
		"max_iterations in CLBFGSMinimizer", MS_NOT_AVAILABLE);
	SG_ADD(&m_delta, "CLBFGSMinimizer__m_delta",
		"delta in CLBFGSMinimizer", MS_NOT_AVAILABLE);
	SG_ADD(&m_past, "CLBFGSMinimizer__m_past",
		"past in CLBFGSMinimizer", MS_NOT_AVAILABLE);
	SG_ADD(&m_epsilon, "CLBFGSMinimizer__m_epsilon",
		"epsilon in CLBFGSMinimizer", MS_NOT_AVAILABLE);
	SG_ADD(&m_min_step, "CLBFGSMinimizer__m_min_step",
		"min_step in CLBFGSMinimizer", MS_NOT_AVAILABLE);
	SG_ADD(&m_max_step, "CLBFGSMinimizer__m_max_step",
		"max_step in CLBFGSMinimizer", MS_NOT_AVAILABLE);
	SG_ADD(&m_ftol, "CLBFGSMinimizer__m_ftol",
		"ftol in CLBFGSMinimizer", MS_NOT_AVAILABLE);
	SG_ADD(&m_wolfe, "CLBFGSMinimizer__m_wolfe",
		"wolfe in CLBFGSMinimizer", MS_NOT_AVAILABLE);
	SG_ADD(&m_gtol, "CLBFGSMinimizer__m_gtol",
		"gtol in CLBFGSMinimizer", MS_NOT_AVAILABLE);
	SG_ADD(&m_xtol, "CLBFGSMinimizer__m_xtol",
		"xtol in CLBFGSMinimizer", MS_NOT_AVAILABLE);
	SG_ADD(&m_orthantwise_c, "CLBFGSMinimizer__m_orthantwise_c",
		"orthantwise_c in CLBFGSMinimizer", MS_NOT_AVAILABLE);
	SG_ADD(&m_orthantwise_start, "CLBFGSMinimizer__m_orthantwise_start",
		"orthantwise_start in CLBFGSMinimizer", MS_NOT_AVAILABLE);
	SG_ADD(&m_orthantwise_end, "CLBFGSMinimizer__m_orthantwise_end",
		"orthantwise_end in CLBFGSMinimizer", MS_NOT_AVAILABLE);
	SG_ADD(&m_target_variable, "CLBFGSMinimizer__m_target_variable",
		"m_target_variable in CLBFGSMinimizer", MS_NOT_AVAILABLE);
}

void CLBFGSMinimizer::set_lbfgs_parameters(
		int32_t m,
		int32_t max_linesearch,
		ELBFGSLineSearch linesearch,
		int32_t max_iterations,
		float64_t delta,
		int32_t past,
		float64_t epsilon,
		float64_t min_step,
		float64_t max_step,
		float64_t ftol,
		float64_t wolfe,
		float64_t gtol,
		float64_t xtol,
		float64_t orthantwise_c,
		int32_t orthantwise_start,
		int32_t orthantwise_end)
{
	m_m = m;
	m_max_linesearch = max_linesearch;
	m_linesearch_id = LBFGSLineSearchHelper::get_lbfgs_linear_search_id(linesearch);
	m_max_iterations = max_iterations;
	m_delta = delta;
	m_past = past;
	m_epsilon = epsilon;
	m_min_step = min_step;
	m_max_step = max_step;
	m_ftol = ftol;
	m_wolfe = wolfe;
	m_gtol = gtol;
	m_xtol = xtol;
	m_orthantwise_c = orthantwise_c;
	m_orthantwise_start = orthantwise_start;
	m_orthantwise_end = orthantwise_end;
}

void CLBFGSMinimizer::init_minimization()
{
	REQUIRE(m_fun, "Cost function not set!\n");
	m_target_variable=m_fun->obtain_variable_reference();
	REQUIRE(m_target_variable.vlen>0,"Target variable from cost function must not empty!\n");
}

float64_t CLBFGSMinimizer::minimize()
{
	lbfgs_parameter_t lbfgs_param;
	lbfgs_param.m = m_m;
	lbfgs_param.max_linesearch = m_max_linesearch;
	lbfgs_param.linesearch = LBFGSLineSearchHelper::get_lbfgs_linear_search(m_linesearch_id);
	lbfgs_param.max_iterations = m_max_iterations;
	lbfgs_param.delta = m_delta;
	lbfgs_param.past = m_past;
	lbfgs_param.epsilon = m_epsilon;
	lbfgs_param.min_step = m_min_step;
	lbfgs_param.max_step = m_max_step;
	lbfgs_param.ftol = m_ftol;
	lbfgs_param.wolfe = m_wolfe;
	lbfgs_param.gtol = m_gtol;
	lbfgs_param.xtol = m_xtol;
	lbfgs_param.orthantwise_c = m_orthantwise_c;
	lbfgs_param.orthantwise_start = m_orthantwise_start;
	lbfgs_param.orthantwise_end = m_orthantwise_end;

	init_minimization();

	float64_t cost=0.0;
	int32_t error_code=lbfgs(m_target_variable.vlen, m_target_variable.vector,
		&cost, CLBFGSMinimizer::evaluate,
		NULL, this, &lbfgs_param);

	if(error_code!=0 && error_code!=LBFGS_ALREADY_MINIMIZED)
	{
		SG_SWARNING("Error(s) happened during L-BFGS optimization (error code:%d)\n",
			error_code);
	}

	return cost;
}

float64_t CLBFGSMinimizer::evaluate(void *obj, const float64_t *variable,
	float64_t *gradient, const int32_t dim, const float64_t step)
{
	/* Note that parameters = parameters_pre_iter - step * gradient_pre_iter */
	CLBFGSMinimizer * obj_prt
		= static_cast<CLBFGSMinimizer *>(obj);

	REQUIRE(obj_prt, "The instance object passed to L-BFGS optimizer should not be NULL\n");

	float64_t cost=obj_prt->m_fun->get_cost();

	if (CMath::is_nan(cost) || CMath::is_infinity(cost))
		return cost;

	//get the gradient wrt variable_new
	SGVector<float64_t> grad=obj_prt->m_fun->get_gradient();
	REQUIRE(grad.vlen==dim,
		"The length of gradient (%d) and the length of variable (%d) do not match\n",
		grad.vlen,dim);

	std::copy(grad.vector,grad.vector+dim,gradient);
	return cost;
}

}
