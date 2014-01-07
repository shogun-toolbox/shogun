/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Roman Votyakov
 * Copyright (C) 2012 Jacob Walker
 */

#include <modelselection/GradientModelSelection.h>

#ifdef HAVE_NLOPT

#include <evaluation/GradientResult.h>
#include <modelselection/ParameterCombination.h>
#include <modelselection/ModelSelectionParameters.h>
#include <machine/Machine.h>
#include <nlopt.h>

using namespace shogun;

#ifndef DOXYGEN_SHOULD_SKIP_THIS

/** structure used for NLopt callback function */
struct nlopt_params
{
	/** pointer to machine evaluation */
	CMachineEvaluation* machine_eval;

	/** pointer to current combination */
	CParameterCombination* current_combination;

	/** pointer to parmeter dictionary */
	CMap<TParameter*, CSGObject*>* parameter_dictionary;

	/** do we want to print the state? */
	bool print_state;
};

/** NLopt callback function wrapper
 *
 * @param n number of parameters
 * @param x vector of parameter values
 * @param grad vector of gradient values with respect to parameter
 * @param func_data data needed for the callback function. In this case, its a
 * nlopt_params
 *
 * @return function value
 */
double nlopt_function(unsigned n, const double* x, double* grad, void* func_data)
{
	nlopt_params* params=(nlopt_params*)func_data;

	CMachineEvaluation* machine_eval=params->machine_eval;
	CParameterCombination* current_combination=params->current_combination;
	CMap<TParameter*, CSGObject*>* parameter_dictionary=params->parameter_dictionary;
	bool print_state=params->print_state;

	index_t offset=0;

	// set parameters from vector x
	for (index_t i=0; i<parameter_dictionary->get_num_elements(); i++)
	{
		CMapNode<TParameter*, CSGObject*>* node=parameter_dictionary->get_node_ptr(i);

		TParameter* param=node->key;
		CSGObject* parent=node->data;

		if (param->m_datatype.m_ctype==CT_VECTOR ||
				param->m_datatype.m_ctype==CT_SGVECTOR)
		{
			REQUIRE(param->m_datatype.m_length_y, "Parameter vector %s has no "
					"length\n", param->m_name)

			for (index_t j=0; j<*(param->m_datatype.m_length_y); j++)
			{

				bool result=current_combination->set_parameter(param->m_name,
						(float64_t)x[offset++],	parent, j);
				 REQUIRE(result, "Parameter %s not found in combination tree\n",
						 param->m_name)
			}
		}
		else
		{
			bool result=current_combination->set_parameter(param->m_name,
					(float64_t)x[offset++], parent);
			REQUIRE(result, "Parameter %s not found in combination tree\n",
					param->m_name)
		}
	}

	// apply current combination to the machine
	CMachine* machine=machine_eval->get_machine();
	current_combination->apply_to_machine(machine);
	SG_UNREF(machine);

	// evaluate the machine
	CEvaluationResult* evaluation_result=machine_eval->evaluate();
	CGradientResult* gradient_result=CGradientResult::obtain_from_generic(
			evaluation_result);
	SG_UNREF(evaluation_result);

	if (print_state)
	{
		gradient_result->print_result();
	}

	// get value of the function, gradients and parameter dictionary
	SGVector<float64_t> value=gradient_result->get_value();
	CMap<TParameter*, SGVector<float64_t> >* gradient=gradient_result->get_gradient();
	CMap<TParameter*, CSGObject*>* gradient_dictionary=
		gradient_result->get_paramter_dictionary();
	SG_UNREF(gradient_result);

	offset=0;

	// set derivative for each parameter from parameter dictionary
	for (index_t i=0; i<parameter_dictionary->get_num_elements(); i++)
	{
		CMapNode<TParameter*, CSGObject*>* node=parameter_dictionary->get_node_ptr(i);

		SGVector<float64_t> derivative;

		for (index_t j=0; j<gradient_dictionary->get_num_elements(); j++)
		{
			CMapNode<TParameter*, CSGObject*>* gradient_node=
				gradient_dictionary->get_node_ptr(j);

			if (gradient_node->data==node->data &&
					!strcmp(gradient_node->key->m_name, node->key->m_name))
			{
				derivative=gradient->get_element(gradient_node->key);
			}
		}

		REQUIRE(derivative.vlen, "Can't find gradient wrt %s parameter!\n",
				node->key->m_name);

		memcpy(grad+offset, derivative.vector, sizeof(double)*derivative.vlen);

		offset+=derivative.vlen;
	}

	SG_UNREF(gradient);
	SG_UNREF(gradient_dictionary);

	return (double)(SGVector<float64_t>::sum(value));
}

#endif /* DOXYGEN_SHOULD_SKIP_THIS */

CGradientModelSelection::CGradientModelSelection() : CModelSelection()
{
	init();
}

CGradientModelSelection::CGradientModelSelection(CMachineEvaluation* machine_eval,
		CModelSelectionParameters* model_parameters)
		: CModelSelection(machine_eval, model_parameters)
{
	init();
}

CGradientModelSelection::~CGradientModelSelection()
{
}

void CGradientModelSelection::init()
{
	m_max_evaluations=1000;
	m_grad_tolerance=1e-6;

	SG_ADD(&m_grad_tolerance, "gradient_tolerance",	"Gradient tolerance",
			MS_NOT_AVAILABLE);
	SG_ADD(&m_max_evaluations, "max_evaluations", "Maximum number of evaluations",
			MS_NOT_AVAILABLE);
}

CParameterCombination* CGradientModelSelection::select_model(bool print_state)
{
	if (!m_model_parameters)
	{
		CMachine* machine=m_machine_eval->get_machine();

		CParameterCombination* current_combination=new CParameterCombination(machine);
		SG_REF(current_combination);

		if (print_state)
		{
			SG_PRINT("Initial combination:\n");
			current_combination->print_tree();
		}

		// get total length of variables
		index_t total_variables=current_combination->get_parameters_length();

		// build parameter->value map
		CMap<TParameter*, SGVector<float64_t> >* argument=
			new CMap<TParameter*, SGVector<float64_t> >();
		current_combination->build_parameter_values_map(argument);

		//  unroll current parameter combination into vector
		SGVector<double> x(total_variables);
		index_t offset=0;

		for (index_t i=0; i<argument->get_num_elements(); i++)
		{
			CMapNode<TParameter*, SGVector<float64_t> >* node=argument->get_node_ptr(i);
			memcpy(x.vector+offset, node->data.vector, sizeof(double)*node->data.vlen);
			offset+=node->data.vlen;
		}

		SG_UNREF(argument);

		// create nlopt object and choose MMA (Method of Moving Asymptotes)
		// optimization algorithm
		nlopt_opt opt=nlopt_create(NLOPT_LD_MMA, total_variables);

		// create lower bound vector (lb=-inf)
		SGVector<double> lower_bound(total_variables);
		lower_bound.set_const(1e-6);

		// create upper bound vector (ub=inf)
		SGVector<double> upper_bound(total_variables);
		upper_bound.set_const(HUGE_VAL);

		// set upper and lower bound
		nlopt_set_lower_bounds(opt, lower_bound.vector);
		nlopt_set_upper_bounds(opt, upper_bound.vector);

		// set maximum number of evaluations
		nlopt_set_maxeval(opt, m_max_evaluations);

		// set absolute argument tolearance
		nlopt_set_xtol_abs1(opt, m_grad_tolerance);
		nlopt_set_ftol_abs(opt, m_grad_tolerance);

		// build parameter->sgobject map from current parameter combination
		CMap<TParameter*, CSGObject*>* parameter_dictionary=
			new CMap<TParameter*, CSGObject*>();
		current_combination->build_parameter_parent_map(parameter_dictionary);

		// nlopt parameters
		nlopt_params params;

		params.current_combination=current_combination;
		params.machine_eval=m_machine_eval;
		params.print_state=print_state;
		params.parameter_dictionary=parameter_dictionary;

		// choose evaluation direction (minimize or maximize objective function)
		if (m_machine_eval->get_evaluation_direction()==ED_MINIMIZE)
		{
			if (print_state)
				SG_PRINT("Minimizing objective function:\n");

			nlopt_set_min_objective(opt, nlopt_function, &params);
		}
		else
		{
			if (print_state)
				SG_PRINT("Maximizing objective function:\n");

			nlopt_set_max_objective(opt, nlopt_function, &params);
		}

		// the minimum objective value, upon return
		double minf;

		// optimize our function
		nlopt_result result=nlopt_optimize(opt, x.vector, &minf);

		REQUIRE(result>0, "NLopt failed while optimizing objective function!\n");

		if (print_state)
		{
			SG_PRINT("Best combination:\n");
			current_combination->print_tree();
		}

		// clean up
		nlopt_destroy(opt);
		SG_UNREF(machine);
		SG_UNREF(parameter_dictionary);

		return current_combination;
	}
	else
	{
		SG_NOTIMPLEMENTED
		return NULL;
	}
}

#endif /* HAVE_NLOPT */
