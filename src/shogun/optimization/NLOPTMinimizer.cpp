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
#include <shogun/optimization/NLOPTMinimizer.h>
#include <shogun/optimization/FirstOrderBoundConstraintsCostFunction.h>
using namespace shogun;

CNLOPTMinimizer::CNLOPTMinimizer()
	:CWrappedMinimizer()
{
	init();
}

CNLOPTMinimizer::~CNLOPTMinimizer()
{
}

CNLOPTMinimizer::CNLOPTMinimizer(CFirstOrderCostFunction *fun)
	:CWrappedMinimizer(fun)
{
	init();
}

void CNLOPTMinimizer::init()
{
#ifdef HAVE_NLOPT
	m_max_iterations=1000;
	m_variable_tolerance=1e-6;
	m_function_tolerance=1e-6;
	m_nlopt_algorithm=NLOPT_LD_LBFGS;

	SG_ADD(&m_max_iterations, "max_iterations", "max_iterations", MS_NOT_AVAILABLE);
	SG_ADD(&m_variable_tolerance, "variable_tolerance", "variable_tolerance", MS_NOT_AVAILABLE);
	SG_ADD(&m_function_tolerance, "function_tolerance", "function_tolerance", MS_NOT_AVAILABLE);
	SG_ADD((int *)&m_nlopt_algorithm, "nlopt_algorithm", "nlopt_algorithm", MS_NOT_AVAILABLE);

	set_nlopt_parameters();
#endif
}

float64_t CNLOPTMinimizer::minimization()
{
#ifdef HAVE_NLOPT
	minimization_init();

	nlopt_opt opt=nlopt_create(m_nlopt_algorithm, m_variable_vec.vlen);

	//add bound constraints
	CFirstOrderBoundConstraintsCostFunction* bound_constraints_fun
		=dynamic_cast<CFirstOrderBoundConstraintsCostFunction *>(m_fun);
	if(bound_constraints_fun)
	{
		CMap<TParameter*, SGVector<float64_t> >* bound=bound_constraints_fun->get_lower_bound();
		if(bound)
		{
			if(bound->get_num_elements()>0)
			{
				SGVector<float64_t> lower_bound(m_variable_vec.vlen);
				copy_in_parameter_order(bound, m_variable, lower_bound.vector, lower_bound.vlen);
				nlopt_set_lower_bounds(opt, lower_bound.vector);
			}
			SG_UNREF(bound);
		}

		bound=bound_constraints_fun->get_upper_bound();
		if(bound)
		{
			if(bound->get_num_elements()>0)
			{
				SGVector<float64_t> upper_bound(m_variable_vec.vlen);
				copy_in_parameter_order(bound, m_variable, upper_bound.vector, upper_bound.vlen);
				nlopt_set_upper_bounds(opt, upper_bound.vector);
			}
			SG_UNREF(bound);
		}
	}

	// set maximum number of evaluations
	nlopt_set_maxeval(opt, m_max_iterations);
	// set absolute argument tolearance
	nlopt_set_xtol_abs1(opt, m_variable_tolerance);
	nlopt_set_ftol_abs(opt, m_function_tolerance);

	nlopt_set_min_objective(opt, CNLOPTMinimizer::nlopt_function, this);

#endif
	// the minimum objective value, upon return
	double cost=0.0;

#ifdef HAVE_NLOPT
	// optimize our function
	nlopt_result error_code=nlopt_optimize(opt, m_variable_vec.vector, &cost);
	if(error_code!=1)
	{
		SG_WARNING("Error(s) happened and NLopt failed during minimization (error code:%d)\n",
			error_code);
	}

	// clean up
	nlopt_destroy(opt);

	if(!m_is_in_place)
		copy_in_parameter_order(m_variable_vec.vector, m_variable_vec.vlen, m_variable, m_variable);
#endif

	return cost;
}

#ifdef HAVE_NLOPT
void CNLOPTMinimizer::set_nlopt_parameters(nlopt_algorithm algorithm,
	float64_t max_iterations,
	float64_t variable_tolerance,
	float64_t function_tolerance)
{
	m_nlopt_algorithm=algorithm;
	m_max_iterations=max_iterations;
	m_variable_tolerance=variable_tolerance;
	m_function_tolerance=function_tolerance;
};

double CNLOPTMinimizer::nlopt_function(unsigned dim, const double* variable, double* gradient,
	void* func_data)
{
	CNLOPTMinimizer* obj_prt=static_cast<CNLOPTMinimizer *>(func_data);
	REQUIRE(obj_prt, "The instance object passed to NLopt optimizer should not be NULL\n");

	//update variable
	if (!obj_prt->m_is_in_place)
		copy_in_parameter_order(variable, dim, obj_prt->m_variable, obj_prt->m_variable);

	//get the gradient wrt variable_new
	CMap<TParameter*, SGVector<float64_t> >* grad=obj_prt->m_fun->get_gradient();
	copy_in_parameter_order(grad,obj_prt->m_variable, gradient, dim);

	SG_UNREF(grad);

	double cost=obj_prt->m_fun->get_cost();
	return cost;
}
#endif
