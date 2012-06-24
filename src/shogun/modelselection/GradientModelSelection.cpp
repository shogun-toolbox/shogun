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
#include <nlopt.h>

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
		printf("%s, %f, %f\n", node->key.string, node->data, x[i]);
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
		//printf("%s, %f\n", node->key.string, node->data);
		grad[i] = node->data;
	}

	best_combination=current_combination;

//	SG_UNREF(result);
//	SG_UNREF(current_combination);

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
/*	for(int i = 0; i < n; i++)
	{
		grad[i] = 0;
	}

	return nlopt_func(n, x, grad, data);*/
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

	//Get Random Combination here



	current_combination = m_model_parameters->get_random_combination();

/*	for (index_t i=0; i<combinations->get_num_elements(); ++i)
	{
		CParameterCombination* current_combination=(CParameterCombination*)
				combinations->get_element(i);
		SG_PRINT("trying combination:\n");
		current_combination->print_tree();

	}*/
//	current_combination=(CParameterCombination*)
//			combinations->


/*
	Parameter* p=new Parameter();

	Parameter* q=new Parameter();

	Parameter* z=new Parameter();

	Parameter* d=new Parameter();

	Parameter* e=new Parameter();


	CModelSelectionParameters* pp = (CModelSelectionParameters*)m_model_parameters->m_child_nodes->get_element(0);
	q->add(&pp->m_sgobject, "Inference Method");
    pp = (CModelSelectionParameters*)pp->m_child_nodes->get_element(0);

	p->add(&pp->m_sgobject, "Likelihood Model");

    pp = (CModelSelectionParameters*)pp->m_child_nodes->get_element(0);

	z->add(&((float64_t*)pp->m_values)[0], "sigma");

    pp = (CModelSelectionParameters*)m_model_parameters->m_child_nodes->get_element(0);
    pp = (CModelSelectionParameters*)pp->m_child_nodes->get_element(1);

	d->add(&pp->m_sgobject, "Kernel");

    pp = (CModelSelectionParameters*)pp->m_child_nodes->get_element(0);

    e->add(&((float64_t*)pp->m_values)[0], "width");

	CParameterCombination* fool = new CParameterCombination(p);
	CParameterCombination* fool2 = new CParameterCombination(q);
	CParameterCombination* fool3 = new CParameterCombination(z);
	CParameterCombination* fool4 = new CParameterCombination(d);
	CParameterCombination* fool5 = new CParameterCombination(e);
for(int i = 0; i < n; i++) lb[i] = 0.01;
	fool4->append_child(fool5);
	fool4->print_tree();
	fool->append_child(fool3);
	fool2->append_child(fool);
	fool2->append_child(fool4);

	current_combination=new CParameterCombination();
	current_combination->append_child(fool2);

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
	//	if (print_state)
	//	{
			SG_PRINT("trying combination:\n");
			current_combination->print_tree();
	//	}

		current_combination->apply_to_modsel_parameter(
				machine->m_model_selection_parameters);

		/* note that this may implicitly lock and unlockthe machine */
		CGradientResult* result = (CGradientResult*)(m_machine_eval->evaluate());


		int n = result->gradient.get_num_elements();

		double* lb = new double[n];
		double* x = new double[n];

		for(int i = 0; i < n; i++) lb[i] = 1e-7;

		SG_SPRINT("%i\n", n);

		for(int i = 0; i < n; i++)
		{
			CMapNode<SGString<char>, float64_t>* node = result->gradient.get_node_ptr(i);
			SG_SPRINT("%s\n", node->key.string);
			SG_SPRINT("%i\n", node->key.slen);
			SG_SPRINT("%f\n", node->data);
			x[i] = *((float64_t*)(current_combination->get_parameter(node->key.string)->m_parameter));
			printf("%f\n", x[i]);
			//current_combination->set_parameter(node->key.string, (float64_t) node->data);
		}

		if (print_state)
			result->print_result();

		if (best_combination)
			SG_UNREF(best_combination);

/*		for(int i = 0; i < n; i++)
		{
			CMapNode<SGString<char>, float64_t>* node = result->gradient.get_node_ptr(i);
			//char* name = node->key.;
			//SG_SPRINT("%s\n", node->key.string);
			SG_SPRINT("%i\n", node->key.slen);
			SG_SPRINT("%f\n", node->data);

			//x[i] = *((float64_t*)(current_combination->get_parameter(name)->m_parameter));
		}*/

		nlopt_opt opt;

		nlopt_package foo;

		foo.best_combination = best_combination;
		foo.current_combination = current_combination;
		foo.m_machine_eval = m_machine_eval;

		nlopt_set_xtol_rel(opt, 1e-4);

		opt = nlopt_create(NLOPT_LD_MMA, n); // algorithm and dimensionality
		nlopt_set_maxeval(opt, 1000);
		nlopt_set_lower_bounds(opt, lb);
		nlopt_set_min_objective(opt, nlopt_function, &foo);

		double minf; //the minimum objective value, upon return

		if (nlopt_optimize(opt, x, &minf) < 0) {
		    printf("nlopt failed!\n");
		}
		else {
		    printf("found minimum at f(%g,%g) = %0.10g\n", x[0], x[1], minf);
		}

		best_combination = foo.best_combination;

//		SG_UNREF(result);
//		SG_UNREF(current_combination);
//	}

//	SG_UNREF(machine);
//	SG_UNREF(combinations);


	return current_combination;
}

}
