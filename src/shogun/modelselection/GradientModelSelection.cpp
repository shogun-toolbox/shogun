/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Jacob Walker
 */

#include <shogun/modelselection/GradientModelSelection.h>
#include <shogun/modelselection/ParameterCombination.h>
#include <shogun/modelselection/ModelSelectionParameters.h>
#include <shogun/machine/Machine.h>
#include <shogun/lib/Map.h>

using namespace shogun;

#ifdef HAVE_NLOPT

#include <nlopt.h>

double CGradientModelSelection::nlopt_function(unsigned n,
		const double *x, double *grad, void *my_func_data)
{
	nlopt_package* pack = (nlopt_package*)my_func_data;

	shogun::CMachineEvaluation* m_machine_eval = pack->m_machine_eval;

	shogun::CParameterCombination* m_current_combination =
			pack->m_current_combination;

	bool print_state = pack->print_state;

	/* Get result vector first to get names of parameters*/
	shogun::CGradientResult* result =
			(shogun::CGradientResult*)(m_machine_eval->evaluate());

	shogun::CMachine* machine=m_machine_eval->get_machine();

	if(print_state)
		result->print_result();

	/*Set parameter values from x vector*/
	for(unsigned int i = 0; i < n; i++)
	{
		shogun::CMapNode<shogun::SGString<char>, float64_t>* node =
				result->gradient.get_node_ptr(i);

		char* name = node->key.string;

		m_current_combination->set_parameter(name, x[i]);
	}

	/*Apply them to the machine*/
	m_current_combination->apply_to_modsel_parameter(
			machine->m_model_selection_parameters);

	/*Get rid of this first result*/
	SG_UNREF(result);

	/*Get a result based on updated parameter values*/
	result = (shogun::CGradientResult*)(m_machine_eval->evaluate());

	/*Store the gradient into the grad vector*/
	for(unsigned int i = 0; i < n; i++)
	{
		shogun::CMapNode<shogun::SGString<char>, float64_t>* node =
				result->gradient.get_node_ptr(i);
		grad[i] = node->data;
	}

	/*Get function value*/
	float64_t function_value = result->quantity[0];

	SG_UNREF(result);
	SG_UNREF(machine);

	return function_value;
}

#endif

CGradientModelSelection::CGradientModelSelection(
		CModelSelectionParameters* model_parameters,
		CMachineEvaluation* machine_eval) : CModelSelection(model_parameters,
				machine_eval) {
	m_max_evaluations = 1000;
	m_grad_tolerance = 1e-4;
	m_current_combination = NULL;
}

CGradientModelSelection::CGradientModelSelection() : CModelSelection(NULL,
		NULL) {
}

CGradientModelSelection::~CGradientModelSelection()
{
	SG_UNREF(m_current_combination);
}

CParameterCombination* CGradientModelSelection::select_model(bool print_state)
{

#ifdef HAVE_NLOPT

	//Get a random initial combination
	SG_UNREF(m_current_combination);
	m_current_combination = m_model_parameters->get_random_combination();
	SG_REF(m_current_combination);

	CMachine* machine=m_machine_eval->get_machine();

	if(print_state)
	{
		SG_PRINT("trying combination:\n");
		m_current_combination->print_tree();
	}

	m_current_combination->apply_to_modsel_parameter(
			machine->m_model_selection_parameters);

	/*How many of these parameters have derivatives?*/
	CGradientResult* result = (CGradientResult*)(m_machine_eval->evaluate());

	int n = result->gradient.get_num_elements();

	double* lb = new double[n];
	double* x = new double[n];

	//Set lower bounds for parameters
	for(int i = 0; i < n; i++) lb[i] = 1e-10;

	//Update x with initial values
	for(int i = 0; i < n; i++)
	{
		CMapNode<SGString<char>, float64_t>* node =
				result->gradient.get_node_ptr(i);

	    TParameter* param =
	    		m_current_combination->get_parameter(node->key.string);

	    if(!param)
	    	SG_ERROR("Could not find parameter %s"\
	    			"in Parameter Combination", node->key.string);

		x[i] = *((float64_t*)(param->m_parameter));
	}

	//Setting up nlopt
	nlopt_opt opt;

	nlopt_package pack;

	pack.m_current_combination = m_current_combination;
	pack.m_machine_eval = m_machine_eval;
	pack.print_state = print_state;

	opt = nlopt_create(NLOPT_LD_MMA, n); // algorithm and dimensionality
	nlopt_set_maxeval(opt, m_max_evaluations);
	nlopt_set_xtol_rel(opt, m_grad_tolerance);
	nlopt_set_lower_bounds(opt, lb);

	//if(m_machine_eval->get_evaluation_direction() == ED_MINIMIZE)
		nlopt_set_min_objective(opt, nlopt_function, &pack);

	//else
	//	nlopt_set_max_objective(opt, nlopt_function, &pack);

	double minf; //the minimum objective value, upon return

	//Optimize our function!
	if (nlopt_optimize(opt, x, &minf) < 0)
	{
		SG_ERROR("nlopt failed!\n");
	}

	//Clean up.
	delete[] lb;
    delete[] x;
    nlopt_destroy(opt);

	//Admittedly weird, but I am unreferencing
	//m_current_combination from this stack and
	//passing it on to another.
	SG_UNREF(machine);
	SG_UNREF(result);
	SG_REF(m_current_combination);

	return m_current_combination;

#endif

	//If we don't have NLOPT then return nothing.
	SG_PRINT("Shogun not configured for NLOPT. Returning NULL combination\n");

	return NULL;
}
