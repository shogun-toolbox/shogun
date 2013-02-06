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

	if (result->get_result_type() != GRADIENTEVALUATION_RESULT)
		SG_SERROR("Evaluation result not a GradientEvaluationResult!")

	if ((unsigned)result->total_variables != n)
	{
		SG_SERROR("Mismatch between total variables in result and variables in " \
				"NLOPT!\n");
	}

	shogun::CMachine* machine = m_machine_eval->get_machine();

	if (print_state)
		result->print_result();

	index_t curr_index = 0;

	/*Set parameter values from x vector*/
	for (index_t i = 0; i < result->gradient.get_num_elements(); i++)
	{
		shogun::CMapNode<TParameter*, SGVector<float64_t> >* node =
				result->gradient.get_node_ptr(i);

		TParameter* param = node->key;

	    CSGObject* parent = result->parameter_dictionary.get_element(param);

	    if (param->m_datatype.m_ctype == CT_VECTOR)
	    {
	    	if (!param->m_datatype.m_length_y)
	    		SG_SERROR("Parameter vector %s has no length\n", param->m_name)

	    	index_t length = *(param->m_datatype.m_length_y);

	    	for (index_t j = 0; j < length; j++)
	    	{
	    		if (!parent || !m_current_combination->set_parameter(
	    				param->m_name, (float64_t)x[curr_index+j], parent, j))
	    		{
	    					SG_SERROR("Parameter %s not found in combination \
	    							tree.\n",
	    							param->m_name);
	    		}
	    	}

	    	curr_index += length;
	    }

	    else if (param->m_datatype.m_ctype == CT_SGVECTOR)
	    {
	    	if (!param->m_datatype.m_length_y)
	    		SG_SERROR("Parameter vector %s has no length\n", param->m_name)

	    	index_t length = *(param->m_datatype.m_length_y);

	    	for (index_t j = 0; j < length; j++)
	    	{
	    		if (!parent || !m_current_combination->set_parameter(
	    				param->m_name, (float64_t)x[curr_index+j], parent, j))
	    		{
	    					SG_SERROR("Parameter %s not found in combination \
	    							tree.\n",
	    							param->m_name);
	    		}
	    	}
	    	curr_index += length;
	    }

	    else
	    {
	    	if (!parent || !m_current_combination->set_parameter(
	    			param->m_name, (float64_t)x[curr_index], parent))
	    	{
	    		SG_SERROR("Parameter %s not found in combination tree.\n",
	    				param->m_name);
	    	}
	    	curr_index++;
	    }
	}

	/*Apply them to the machine*/
	m_current_combination->apply_to_modsel_parameter(
			machine->m_model_selection_parameters);

	/*Get rid of this first result*/
	SG_UNREF(result);

	/*Get a result based on updated parameter values*/
	result = (shogun::CGradientResult*)(m_machine_eval->evaluate());

	if (result->get_result_type() != GRADIENTEVALUATION_RESULT)
		SG_SERROR("Evaluation result not a GradientEvaluationResult!")

	curr_index = 0;

	/*Store the gradient into the grad vector*/
	for (index_t i = 0; i < result->gradient.get_num_elements(); i++)
	{
		shogun::CMapNode<TParameter*, SGVector<float64_t> >* node =
				result->gradient.get_node_ptr(i);

		for(index_t j = 0; j < node->data.vlen; j++)
			grad[curr_index+j] = node->data[j];

		curr_index += node->data.vlen;
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
	init();
}

void CGradientModelSelection::init()
{
	m_max_evaluations = 1000;
	m_grad_tolerance = 1e-4;
	m_current_combination = NULL;

	SG_ADD((CSGObject**)&m_current_combination, "current_combination",
			"Current Combination", MS_NOT_AVAILABLE);
	SG_ADD(&m_grad_tolerance, "gradient_tolerance",
			"gradient_tolerance", MS_NOT_AVAILABLE);
	SG_ADD(&m_max_evaluations, "max_evaluations", "Max Evaluations",
			MS_NOT_AVAILABLE);
}

CGradientModelSelection::CGradientModelSelection() : CModelSelection(NULL,
		NULL)
{
	init();
}

CGradientModelSelection::~CGradientModelSelection()
{
	SG_UNREF(m_current_combination);
}

void CGradientModelSelection::test_gradients()
{

	float64_t delta = 0.001;
	float64_t error_tol = 0.1;
	float64_t orig_value, new_value;
	float64_t orig_eval, new_eval;
	float64_t approx_grad, true_grad;

	CMachine* machine = m_machine_eval->get_machine();

	m_current_combination->apply_to_modsel_parameter(
				machine->m_model_selection_parameters);

	CGradientResult* result = (CGradientResult*)(m_machine_eval->evaluate());

	/*Set parameter values from x vector*/
	for (index_t i = 0; i < result->gradient.get_num_elements(); i++)
	{
		shogun::CMapNode<TParameter*, SGVector<float64_t> >* node =
				result->gradient.get_node_ptr(i);

		orig_eval = result->quantity[0];

		TParameter* param = node->key;

		for (index_t j = 0; j < node->data.vlen; j++)
		{
			true_grad = node->data[j];

			index_t index = -1;

			if (param->m_datatype.m_ctype == CT_VECTOR ||
				param->m_datatype.m_ctype == CT_SGVECTOR)
			{
				index = j;
				orig_value = (*((float64_t**)param->m_parameter))[j];
			}

			else
				orig_value = *((float64_t*)param->m_parameter);

			new_value = orig_value+delta;

		    CSGObject* parent = result->parameter_dictionary.get_element(param);


		    if (!parent || !m_current_combination->set_parameter(
		    		param->m_name, new_value, parent, index))
		    {
				SG_ERROR("Parameter %s not found in combination tree.\n",
						param->m_name);
		    }

			m_current_combination->apply_to_modsel_parameter(
					machine->m_model_selection_parameters);

			CGradientResult* new_result =
					(CGradientResult*)(m_machine_eval->evaluate());

			new_eval = new_result->quantity[0];

			approx_grad = (new_eval-orig_eval)/delta;

			if (abs(approx_grad - true_grad) > error_tol)
			{
				SG_ERROR("Gradient of function with respect to variable %i of %s" \
						  " is incorrect.\n" \
						  "True value is approximately %f, but calculated value is" \
						  " %f\n", j, param->m_name,
						  approx_grad, true_grad);
			}

			if (!parent || !m_current_combination->set_parameter(
		    		param->m_name, orig_value, parent, index))
			{
				SG_ERROR("Parameter %s not found in combination tree.\n",
						param->m_name);
			}

			m_current_combination->apply_to_modsel_parameter(
					machine->m_model_selection_parameters);

			SG_UNREF(new_result);
		}
	}

	SG_UNREF(machine);
	SG_UNREF(result);
}

CParameterCombination* CGradientModelSelection::select_model(bool print_state)
{

#ifdef HAVE_NLOPT

	//Get a random initial combination
	SG_UNREF(m_current_combination);
	m_current_combination = m_model_parameters->get_single_combination();
	SG_REF(m_current_combination);

	CMachine* machine = m_machine_eval->get_machine();

	if (print_state)
	{
		SG_PRINT("trying combination:\n")
		m_current_combination->print_tree();
	}

	m_current_combination->apply_to_modsel_parameter(
			machine->m_model_selection_parameters);

	/*How many of these parameters have derivatives?*/
	CGradientResult* result = (CGradientResult*)(m_machine_eval->evaluate());

	if (result->get_result_type() != GRADIENTEVALUATION_RESULT)
		SG_ERROR("Evaluation result not a GradientEvaluationResult!")

	index_t n = result->total_variables;

	double* lb = SG_MALLOC(double, n);
	double* x = SG_MALLOC(double, n);

	index_t cur_index = 0;

	//Update x with initial values
	for (index_t i = 0; i < result->gradient.get_num_elements(); i++)
	{
		shogun::CMapNode<TParameter*, SGVector<float64_t> >* node =
				result->gradient.get_node_ptr(i);

		TParameter* param = node->key;

	    CSGObject* parent = result->parameter_dictionary.get_element(param);

	    TParameter* final = m_current_combination->get_parameter(
	    		param->m_name, parent);

	    if (!final)
	    {
	    	SG_ERROR("Could not find parameter %s "\
	    			"in Parameter Combination\n", param->m_name);
	    }

	    if (final->m_datatype.m_ctype == CT_VECTOR)
	    {
	    	if (!param->m_datatype.m_length_y)
	    		SG_ERROR("Parameter vector %s has no length\n", param->m_name)

	    	index_t length = *(final->m_datatype.m_length_y);

	    	for (index_t j = 0; j < length; j++)
	    		x[cur_index+j] = *((float64_t**)(final->m_parameter))[j];

	    	cur_index += length;
	    }

	    else if (final->m_datatype.m_ctype == CT_SGVECTOR)
	    {
	    	if (!param->m_datatype.m_length_y)
	    		SG_ERROR("Parameter vector %s has no length\n", param->m_name)

	    	index_t length = *(final->m_datatype.m_length_y);

	    	for (index_t j = 0; j < length; j++)
	    		x[cur_index+j] = (*(float64_t**)(final->m_parameter))[j];

	    	cur_index += length;
	    }

	    else
	    {
	    	x[cur_index] = *((float64_t*)(final->m_parameter));
	    	cur_index++;
	    }

	}

	cur_index = 0;

	CParameterCombination* lower_combination =
			m_model_parameters->get_single_combination(false);

	//Update x with initial values
	for (index_t i = 0; i < result->gradient.get_num_elements(); i++)
	{
		shogun::CMapNode<TParameter*, SGVector<float64_t> >* node =
				result->gradient.get_node_ptr(i);

		TParameter* param = node->key;

	    CSGObject* parent = result->parameter_dictionary.get_element(param);

	    TParameter* final = lower_combination->get_parameter(
	    		param->m_name, parent);

	    if (!final)
	    {
	    	SG_ERROR("Could not find parameter %s "\
	    			"in Parameter Combination\n", param->m_name);
	    }

	    if (final->m_datatype.m_ctype == CT_VECTOR)
	    {
	    	if (!param->m_datatype.m_length_y)
	    		SG_ERROR("Parameter vector %s has no length\n", param->m_name)

	    	index_t length = *(final->m_datatype.m_length_y);

	    	for (index_t j = 0; j < length; j++)
	    		lb[cur_index+j] = *((float64_t**)(final->m_parameter))[j];

	    	cur_index += length;
	    }

	    else if (final->m_datatype.m_ctype == CT_SGVECTOR)
	    {
	    	if (!param->m_datatype.m_length_y)
	    		SG_ERROR("Parameter vector %s has no length\n", param->m_name)

	    	index_t length = *(final->m_datatype.m_length_y);

	    	for (index_t j = 0; j < length; j++)
	    		lb[cur_index+j] = (*(float64_t**)(final->m_parameter))[j];

	    	cur_index += length;
	    }

	    else
	    {
	    	lb[cur_index] = *((float64_t*)(final->m_parameter));
	    	cur_index++;
	    }

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

	if (m_machine_eval->get_evaluation_direction() == ED_MINIMIZE)
	{
		if (print_state)
			SG_SPRINT("Minimizing Objective Function\n")

		nlopt_set_min_objective(opt, nlopt_function, &pack);
	}

	else
	{
		if (print_state)
			SG_SPRINT("Maximizing Objective Function\n")

		nlopt_set_max_objective(opt, nlopt_function, &pack);
	}

	double minf; //the minimum objective value, upon return

//	test_gradients();

	//Optimize our function!
	if (nlopt_optimize(opt, x, &minf) < 0)
		SG_ERROR("nlopt failed!\n")

//	test_gradients();

	//Clean up.
	SG_FREE(lb);
	SG_FREE(x);
	nlopt_destroy(opt);

	//Admittedly weird, but I am unreferencing
	//m_current_combination from this stack and
	//passing it on to another.
	SG_UNREF(machine);
	SG_UNREF(result);
	SG_UNREF(lower_combination);

	SG_REF(m_current_combination);

	return m_current_combination;

#endif

	//If we don't have NLOPT then return nothing.
	SG_PRINT("Shogun not configured for NLOPT. Returning NULL combination\n")

	return NULL;
}
