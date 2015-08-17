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
#include <algorithm>
#include <shogun/optimization/NLOPTMinimizer.h>
#include <shogun/optimization/FirstOrderBoundConstraintsCostFunction.h>
using namespace shogun;

NLOPTMinimizer::NLOPTMinimizer()
	:FirstOrderMinimizer()
{
	init();
}

NLOPTMinimizer::~NLOPTMinimizer()
{
}

NLOPTMinimizer::NLOPTMinimizer(FirstOrderCostFunction *fun)
	:FirstOrderMinimizer(fun)
{
	init();
}

void NLOPTMinimizer::init()
{
#ifdef HAVE_NLOPT
	m_max_iterations=1000;
	m_variable_tolerance=1e-6;
	m_function_tolerance=1e-6;
	m_nlopt_algorithm=NLOPT_LD_LBFGS;
	m_target_variable=SGVector<float64_t>();
	set_nlopt_parameters();
#endif
}

float64_t NLOPTMinimizer::minimize()
{
#ifdef HAVE_NLOPT
	init_minimization();

	nlopt_opt opt=nlopt_create(m_nlopt_algorithm, m_target_variable.vlen);

	//add bound constraints
	FirstOrderBoundConstraintsCostFunction* bound_constraints_fun
		=dynamic_cast<FirstOrderBoundConstraintsCostFunction *>(m_fun);
	if(bound_constraints_fun)
	{
		SGVector<float64_t> bound=bound_constraints_fun->get_lower_bound();
		if(bound.vlen==1)
		{
			nlopt_set_lower_bounds1(opt, bound[0]);
		}
		else
		{
			REQUIRE(bound.vlen==m_target_variable.vlen,
				"The length of target variable (%d) and the length of lower bound (%d) do not match\n",
				m_target_variable.vlen, bound.vlen);
			nlopt_set_lower_bounds(opt, bound.vector);
		}

		bound=bound_constraints_fun->get_upper_bound();
		if(bound.vlen==1)
		{
			nlopt_set_upper_bounds1(opt, bound[0]);
		}
		else
		{
			REQUIRE(bound.vlen==m_target_variable.vlen,
			"The length of target variable (%d) and the length of upper bound (%d) do not match\n",
				m_target_variable.vlen, bound.vlen);
			nlopt_set_upper_bounds(opt, bound.vector);
		}
	
	}
	// set maximum number of evaluations
	nlopt_set_maxeval(opt, m_max_iterations);
	// set absolute argument tolearance
	nlopt_set_xtol_abs1(opt, m_variable_tolerance);
	nlopt_set_ftol_abs(opt, m_function_tolerance);

	nlopt_set_min_objective(opt, NLOPTMinimizer::nlopt_function, this);

#endif
	// the minimum objective value, upon return
	double cost=0.0;

#ifdef HAVE_NLOPT
	// optimize our function
	nlopt_result error_code=nlopt_optimize(opt, m_target_variable.vector, &cost);
	if(error_code!=1)
	{
		SG_SWARNING("Error(s) happened and NLopt failed during minimization (error code:%d)\n",
			error_code);
	}

	// clean up
	nlopt_destroy(opt);
#endif

	return cost;
}

#ifdef HAVE_NLOPT
void NLOPTMinimizer::set_nlopt_parameters(nlopt_algorithm algorithm,
	float64_t max_iterations,
	float64_t variable_tolerance,
	float64_t function_tolerance)
{
	m_nlopt_algorithm=algorithm;
	m_max_iterations=max_iterations;
	m_variable_tolerance=variable_tolerance;
	m_function_tolerance=function_tolerance;
};

double NLOPTMinimizer::nlopt_function(unsigned dim, const double* variable, double* gradient,
	void* func_data)
{
	NLOPTMinimizer* obj_prt=static_cast<NLOPTMinimizer *>(func_data);
	REQUIRE(obj_prt, "The instance object passed to NLopt optimizer should not be NULL\n");

	//get the gradient wrt variable_new
	SGVector<float64_t> grad=obj_prt->m_fun->get_gradient();

	REQUIRE(grad.vlen==(index_t)dim,
		"The length of gradient (%d) and the length of variable (%d) do not match\n",
		grad.vlen,dim);

	std::copy(grad.vector,grad.vector+dim,gradient);

	double cost=obj_prt->m_fun->get_cost();
	return cost;
}

void NLOPTMinimizer::init_minimization()
{
	REQUIRE(m_fun, "Cost function not set!\n");
	m_target_variable=m_fun->obtain_variable_reference();
	REQUIRE(m_target_variable.vlen>0,"Target variable from cost function must not empty!\n");
}
#endif
