/*
 * GradientModelSelection.cpp
 *
 *  Created on: Jun 15, 2012
 *      Author: jacobw
 */

#include <shogun/modelselection/GradientModelSelection.h>
#include <shogun/modelselection/ParameterCombination.h>
#include <shogun/modelselection/ModelSelectionParameters.h>
#include <shogun/machine/Machine.h>
#include <shogun/lib/Map.h>

struct nlopt_package
{
	shogun::CMachineEvaluation* m_machine_eval;
	shogun::CParameterCombination* current_combination;
	shogun::CParameterCombination* best_combination;
};

double nlopt_function(unsigned n, const double *x, double *grad, void *my_func_data)
{
	nlopt_package* pack = (nlopt_package*)my_func_data;

	shogun::CMachineEvaluation* m_machine_eval = pack->m_machine_eval;
	shogun::CParameterCombination* current_combination = pack->current_combination;
	shogun::CParameterCombination* best_combination = pack->best_combination;

	/* note that this may implicitly lock and unlockthe machine */
	shogun::CGradientResult* result = (shogun::CGradientResult*)(m_machine_eval->evaluate());

	shogun::CMachine* machine=m_machine_eval->get_machine();

	for(int i = 0; i < n; i++)
	{
		shogun::CMapNode<shogun::SGString<char>, float64_t>* node = result->gradient.get_node_ptr(i);
		char* name = node->key.string;
		current_combination->set_parameter(name, x[i]);
	}

	current_combination->apply_to_modsel_parameter(
					machine->m_model_selection_parameters);

			/* note that this may implicitly lock and unlockthe machine */
	SG_UNREF(result);

	result = (shogun::CGradientResult*)(m_machine_eval->evaluate());

	for(int i = 0; i < n; i++)
	{
		shogun::CMapNode<shogun::SGString<char>, float64_t>* node = result->gradient.get_node_ptr(i);
		grad[i] = node->data;
	}

	best_combination=current_combination;

	SG_UNREF(result);
	SG_UNREF(current_combination);

	return result->quantity[0];
}




namespace shogun {

CGradientModelSelection::CGradientModelSelection(CModelSelectionParameters* model_parameters,
		CMachineEvaluation* machine_eval) : CModelSelection(model_parameters,
				machine_eval) {
	// TODO Auto-generated constructor stub

}

double CGradientModelSelection::nlopt_const(unsigned n, const double *x, double *grad, void *data)
{
	/*for(int i = 0; i < n; i++)
	{
		grad[i] = 0;
	}

	return nlopt_func(n, x, grad, NULL);*/
}

CGradientModelSelection::CGradientModelSelection() : CModelSelection(NULL,
				NULL) {
	// TODO Auto-generated constructor stub

}

CGradientModelSelection::~CGradientModelSelection() {
	// TODO Auto-generated destructor stub
}

CParameterCombination* CGradientModelSelection::select_model(bool print_state)
{
	int num_iterations = 1000;

	//Get Random Combination here



	CDynamicObjectArray* combinations=
			(CDynamicObjectArray*)m_model_parameters->get_random_combination();

	current_combination=(CParameterCombination*)
			combinations->get_last_element();

	best_combination=NULL;
	/*if (m_machine_eval->get_evaluation_direction()==ED_MAXIMIZE)
		b=CMath::ALMOST_NEG_INFTY;
	else
		best_result.mean=CMath::ALMOST_INFTY;*/

	CMachine* machine=m_machine_eval->get_machine();


	/* underlying learning machine */

	/* apply all combinations and search for best one */
//	for (index_t i=0; i<num_iterations; ++i)
//	{

		/* eventually print */
		if (print_state)
		{
			SG_PRINT("trying combination:\n");
			current_combination->print_tree();
		}

		current_combination->apply_to_modsel_parameter(
				machine->m_model_selection_parameters);

		/* note that this may implicitly lock and unlockthe machine */
		CGradientResult* result = (CGradientResult*)(m_machine_eval->evaluate());

		if (print_state)
			result->print_result();

		if (best_combination)
			SG_UNREF(best_combination);

		best_combination=current_combination;

		best_result = (*result);

		int n = result->gradient.get_num_elements();
/*
		double* lb = new double[n];
		double* x = new double[n];

		for(int i = 0; i < n; i++) lb[i] = 0;

		for(int i = 0; i < n; i++)
		{
			CMapNode<SGString<char>, float64_t>* node = result->gradient.get_node_ptr(i);
			char* name = node->key.string;
			x[i] = *((float64_t*)(current_combination->get_parameter(name)->m_parameter));
		}

		nlopt_opt opt;

		opt = nlopt_create(NLOPT_LD_MMA, n); /* algorithm and dimensionality 
		nlopt_set_lower_bounds(opt, lb);
		nlopt_set_min_objective(opt, nlopt_function, NULL);

		double minf; the minimum objective value, upon return 

		if (nlopt_optimize(opt, x, &minf) < 0) {
		    printf("nlopt failed!\n");
		}
		else {
		    printf("found minimum at f(%g,%g) = %0.10g\n", x[0], x[1], minf);
		}*/


		SG_UNREF(result);
		SG_UNREF(current_combination);
//	}

	SG_UNREF(machine);
	SG_UNREF(combinations);


	return best_combination;
}

}
