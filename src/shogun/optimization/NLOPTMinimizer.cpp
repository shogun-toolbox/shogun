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

#include <shogun/optimization/NLOPTMinimizer.h>
#include <shogun/optimization/FirstOrderBoundConstraintsCostFunction.h>
#include <shogun/base/Parameter.h>
#include <algorithm> 

using namespace shogun;
#ifdef USE_GPL_SHOGUN
CNLOPTMinimizer::CNLOPTMinimizer()
	:FirstOrderMinimizer()
{
	init();
}

CNLOPTMinimizer::~CNLOPTMinimizer()
{
}

CNLOPTMinimizer::CNLOPTMinimizer(FirstOrderCostFunction *fun)
	:FirstOrderMinimizer(fun)
{
	init();
}

void CNLOPTMinimizer::init()
{
#ifdef HAVE_NLOPT
	m_target_variable=SGVector<float64_t>();
	set_nlopt_parameters();
	SG_ADD(&m_max_iterations, "CNLOPTMinimizer__m_max_iterations",
		"max_iterations in CNLOPTMinimizer", MS_NOT_AVAILABLE);
	SG_ADD(&m_variable_tolerance, "CNLOPTMinimizer__m_variable_tolerance",
		"variable_tolerance in CNLOPTMinimizer", MS_NOT_AVAILABLE);
	SG_ADD(&m_function_tolerance, "CNLOPTMinimizer__m_function_tolerance",
		"function_tolerance in CNLOPTMinimizer", MS_NOT_AVAILABLE);
	SG_ADD(&m_nlopt_algorithm_id, "CNLOPTMinimizer__m_nlopt_algorithm_id",
		"nlopt_algorithm_id in CNLOPTMinimizer", MS_NOT_AVAILABLE);
#endif
}

float64_t CNLOPTMinimizer::minimize()
{
#ifdef HAVE_NLOPT
	init_minimization();

	nlopt_opt opt=nlopt_create(get_nlopt_algorithm(m_nlopt_algorithm_id),
		m_target_variable.vlen);

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
		else if (bound.vlen>1)
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
		else if (bound.vlen>1)
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

	nlopt_set_min_objective(opt, CNLOPTMinimizer::nlopt_function, this);

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
int16_t CNLOPTMinimizer::get_nlopt_algorithm_id(ENLOPTALGORITHM method)
{
	int16_t method_id=-1;
	switch(method)
	{
	case  GN_DIRECT:
		method_id = (int16_t) NLOPT_GN_DIRECT;
		break; 
	case  GN_DIRECT_L:
		method_id = (int16_t) NLOPT_GN_DIRECT_L;
		break; 
	case  GN_DIRECT_L_RAND:
		method_id = (int16_t) NLOPT_GN_DIRECT_L_RAND;
		break; 
	case  GN_DIRECT_NOSCAL:
		method_id = (int16_t) NLOPT_GN_DIRECT_NOSCAL;
		break; 
	case  GN_DIRECT_L_NOSCAL:
		method_id = (int16_t) NLOPT_GN_DIRECT_L_NOSCAL;
		break; 
	case  GN_DIRECT_L_RAND_NOSCAL:
		method_id = (int16_t) NLOPT_GN_DIRECT_L_RAND_NOSCAL;
		break; 
	case  GN_ORIG_DIRECT:
		method_id = (int16_t) NLOPT_GN_ORIG_DIRECT;
		break; 
	case  GN_ORIG_DIRECT_L:
		method_id = (int16_t) NLOPT_GN_ORIG_DIRECT_L;
		break; 
	case  GN_CRS2_LM:
		method_id = (int16_t) NLOPT_GN_CRS2_LM;
		break; 
	case  GN_ISRES:
		method_id = (int16_t) NLOPT_GN_ISRES;
		break; 
	case  LD_MMA:
		method_id = (int16_t) NLOPT_LD_MMA;
		break; 
	case  LD_LBFGS:
		method_id = (int16_t) NLOPT_LD_LBFGS;
		break; 
	case  LD_LBFGS_NOCEDAL:
		method_id = (int16_t) NLOPT_LD_LBFGS_NOCEDAL;
		break; 
	case  LD_VAR1:
		method_id = (int16_t) NLOPT_LD_VAR1;
		break; 
	case  LD_VAR2:
		method_id = (int16_t) NLOPT_LD_VAR2;
		break; 
	case  LD_TNEWTON:
		method_id = (int16_t) NLOPT_LD_TNEWTON;
		break; 
	case  LD_TNEWTON_RESTART:
		method_id = (int16_t) NLOPT_LD_TNEWTON_RESTART;
		break; 
	case  LD_TNEWTON_PRECOND:
		method_id = (int16_t) NLOPT_LD_TNEWTON_PRECOND;
		break; 
	case  LD_TNEWTON_PRECOND_RESTART:
		method_id = (int16_t) NLOPT_LD_TNEWTON_PRECOND_RESTART;
		break; 
	case  LD_SLSQP:
		method_id = (int16_t) NLOPT_LD_SLSQP;
		break; 
	case  LN_PRAXIS:
		method_id = (int16_t) NLOPT_LN_PRAXIS;
		break; 
	case  LN_COBYLA:
		method_id = (int16_t) NLOPT_LN_COBYLA;
		break; 
	case  LN_NEWUOA:
		method_id = (int16_t) NLOPT_LN_NEWUOA;
		break; 
	case  LN_NEWUOA_BOUND:
		method_id = (int16_t) NLOPT_LN_NEWUOA_BOUND;
		break; 
	case  LN_NELDERMEAD:
		method_id = (int16_t) NLOPT_LN_NELDERMEAD;
		break; 
	case  LN_SBPLX:
		method_id = (int16_t) NLOPT_LN_SBPLX;
		break; 
	case  LN_BOBYQA:
		method_id = (int16_t) NLOPT_LN_BOBYQA;
		break; 
	case  AUGLAG:
		method_id = (int16_t) NLOPT_AUGLAG;
		break; 
	case  AUGLAG_EQ:
		method_id = (int16_t) NLOPT_AUGLAG_EQ;
		break; 
	case  G_MLSL:
		method_id = (int16_t) NLOPT_G_MLSL;
		break; 
	case G_MLSL_LDS:
		method_id = (int16_t) NLOPT_G_MLSL_LDS;
		break; 
	};
	REQUIRE(method_id>=0, "Unsupported algorithm\n");
	return method_id;
}

void CNLOPTMinimizer::set_nlopt_parameters(ENLOPTALGORITHM algorithm,
	float64_t max_iterations,
	float64_t variable_tolerance,
	float64_t function_tolerance)
{
	m_nlopt_algorithm_id=get_nlopt_algorithm_id(algorithm);
	m_max_iterations=max_iterations;
	m_variable_tolerance=variable_tolerance;
	m_function_tolerance=function_tolerance;
};

double CNLOPTMinimizer::nlopt_function(unsigned dim, const double* variable, double* gradient,
	void* func_data)
{
	CNLOPTMinimizer* obj_prt=static_cast<CNLOPTMinimizer *>(func_data);
	REQUIRE(obj_prt, "The instance object passed to NLopt optimizer should not be NULL\n");
	REQUIRE((index_t)dim==(obj_prt->m_target_variable).vlen, "Length must be matched\n");

	double *var = const_cast<double *>(variable);
	std::swap_ranges(var, var+dim, (obj_prt->m_target_variable).vector);

	double cost=obj_prt->m_fun->get_cost();

	//get the gradient wrt variable_new
	SGVector<float64_t> grad=obj_prt->m_fun->get_gradient();

	REQUIRE(grad.vlen==(index_t)dim,
		"The length of gradient (%d) and the length of variable (%d) do not match\n",
		grad.vlen,dim);

	std::copy(grad.vector,grad.vector+dim,gradient);

	std::swap_ranges(var, var+dim, (obj_prt->m_target_variable).vector);
	return cost;
}

void CNLOPTMinimizer::init_minimization()
{
	REQUIRE(m_fun, "Cost function not set!\n");
	m_target_variable=m_fun->obtain_variable_reference();
	REQUIRE(m_target_variable.vlen>0,"Target variable from cost function must not empty!\n");
}
#endif

#endif //USE_GPL_SHOGUN
