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
#include <shogun/lib/config.h>
#include <shogun/optimization/lbfgs/LBFGSMinimizer.h>

using namespace shogun;

CLBFGSMinimizer::CLBFGSMinimizer()
	:CWrappedMinimizer()
{
	init();
}

CLBFGSMinimizer::~CLBFGSMinimizer()
{
}

CLBFGSMinimizer::CLBFGSMinimizer(CFirstOrderCostFunction *fun)
	:CWrappedMinimizer(fun)
{
	init();
}

void CLBFGSMinimizer::init()
{
	set_lbfgs_parameters();
	SG_ADD(&m_m, "m",
		"The number of corrections to approximate the inverse Hessian matrix",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_max_linesearch, "max_linesearch",
		"The maximum number of trials to do line search for each L-BFGS update",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_linesearch, "linesearch",
		"The line search algorithm",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_max_iterations, "max_iterations",
		"The maximum number of iterations for L-BFGS update",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_delta, "delta",
		"Delta for convergence test based on the change of function value",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_past, "past",
		"Distance for delta-based convergence test",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_epsilon, "epsilon",
		"Epsilon for convergence test based on the change of gradient",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_min_step, "min_step",
		"The minimum step of the line search",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_max_step, "max_step",
		"The maximum step of the line search",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_ftol, "ftol",
		"A parameter used in Armijo condition",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_wolfe, "wolfe",
		"A parameter used in curvature condition",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_gtol, "gtol",
		"A parameter used in Morethuente linesearch to control the accuracy",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_xtol, "xtol",
		"The machine precision for floating-point values",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_orthantwise_c, "orthantwise_c",
		"Coeefficient for the L1 norm of variables",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_orthantwise_start, "orthantwise_start",
		"Start index for computing L1 norm of the variables",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_orthantwise_end, "orthantwise_end",
		"End index for computing L1 norm of the variables",
		MS_NOT_AVAILABLE);
}

void CLBFGSMinimizer::set_lbfgs_parameters(
		int m,
		int max_linesearch,
		ELBFGSLineSearch linesearch,
		int max_iterations,
		float64_t delta,
		int past,
		float64_t epsilon,
		float64_t min_step,
		float64_t max_step,
		float64_t ftol,
		float64_t wolfe,
		float64_t gtol,
		float64_t xtol,
		float64_t orthantwise_c,
		int orthantwise_start,
		int orthantwise_end)
{
	m_m = m;
	m_max_linesearch = max_linesearch;
	m_linesearch = linesearch;
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

float64_t CLBFGSMinimizer::minimization()
{
	lbfgs_parameter_t lbfgs_param;
	lbfgs_param.m = m_m;
	lbfgs_param.max_linesearch = m_max_linesearch;
	lbfgs_param.linesearch = m_linesearch;
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

	minimization_init();

	float64_t cost=0.0;
	int error_code=lbfgs(m_variable_vec.vlen, m_variable_vec.vector,
		&cost, CLBFGSMinimizer::evaluate,
		NULL, this, &lbfgs_param);

	if(error_code!=0 && error_code!=LBFGS_ALREADY_MINIMIZED)
	{
		SG_WARNING("Error(s) happened during L-BFGS optimization (error code:%d)\n",
			error_code);
	}

	if(!m_is_in_place)
		copy_in_parameter_order(m_variable_vec.vector, m_variable_vec.vlen, m_variable, m_variable);

	return cost;
}

float64_t CLBFGSMinimizer::evaluate(void *obj, const float64_t *variable,
	float64_t *gradient, const int dim, const float64_t step)
{
	/* Note that parameters = parameters_pre_iter - step * gradient_pre_iter */
	CLBFGSMinimizer * obj_prt
		= static_cast<CLBFGSMinimizer *>(obj);

	REQUIRE(obj_prt, "The instance object passed to L-BFGS optimizer should not be NULL\n");

	//update variable
	if (!obj_prt->m_is_in_place)
		copy_in_parameter_order(variable, dim, obj_prt->m_variable, obj_prt->m_variable);

	//get the gradient wrt variable_new
	CMap<TParameter*, SGVector<float64_t> >* grad=obj_prt->m_fun->get_gradient();
	copy_in_parameter_order(grad,obj_prt->m_variable, gradient, dim);

	SG_UNREF(grad);
	float64_t cost=obj_prt->m_fun->get_cost();
	return cost;
}
