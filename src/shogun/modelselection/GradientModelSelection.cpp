/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Jacob Walker, Wu Lin, Heiko Strathmann, Roman Votyakov, Viktor Gal,
 *          Weijie Lin, Fernando Iglesias, Sergey Lisitsyn, Bjoern Esser,
 *          Soeren Sonnenburg
 */

#include <shogun/modelselection/GradientModelSelection.h>

#include <shogun/base/progress.h>
#include <shogun/evaluation/GradientResult.h>
#include <shogun/machine/Machine.h>
#include <shogun/mathematics/Math.h>
#include <shogun/modelselection/ModelSelectionParameters.h>
#include <shogun/modelselection/ParameterCombination.h>
#include <shogun/optimization/FirstOrderCostFunction.h>
#include <shogun/optimization/lbfgs/LBFGSMinimizer.h>

using namespace shogun;

namespace shogun
{
#ifndef DOXYGEN_SHOULD_SKIP_THIS

class GradientModelSelectionCostFunction: public FirstOrderCostFunction
{
public:
	GradientModelSelectionCostFunction():FirstOrderCostFunction() {  init(); }
	virtual ~GradientModelSelectionCostFunction() {  }
	void set_target(std::shared_ptr<GradientModelSelection >obj)
	{
		REQUIRE(obj,"Obj must set\n");
		if(m_obj!=obj)
		{


			m_obj=obj;
		}
	}
	void unset_target(bool is_unref)
	{
		if(is_unref)
		{

		}
		m_obj=NULL;
	}

	virtual float64_t get_cost()
	{
		REQUIRE(m_obj,"Object not set\n");
		return m_obj->get_cost(m_val, m_grad, m_func_data);
	}

	virtual SGVector<float64_t> obtain_variable_reference()
	{
		REQUIRE(m_obj,"Object not set\n");
		return m_val;
	}
	virtual SGVector<float64_t> get_gradient()
	{
		REQUIRE(m_obj,"Object not set\n");
		return m_grad;
	}

	virtual const char* get_name() const { return "GradientModelSelectionCostFunction"; }

	virtual void set_func_data(void *func_data)
	{
		REQUIRE(func_data != NULL, "func_data must set\n");
		m_func_data = func_data;
	}

	virtual void set_variables(SGVector<float64_t> val)
	{
		m_val = SGVector<float64_t>(val.vlen);
		m_grad = SGVector<float64_t>(val.vlen);
		std::copy(val.vector,val.vector+val.vlen,m_val.vector);
	}
private:
	void init()
	{
		m_obj=NULL;
		SG_ADD((std::shared_ptr<SGObject>*)&m_obj, "GradientModelSelectionCostFunction__m_obj",
			"obj in GradientModelSelectionCostFunction");
		m_func_data = NULL;
		m_val = SGVector<float64_t>();
		SG_ADD(
			&m_val, "GradientModelSelectionCostFunction__m_val",
			"val in GradientModelSelectionCostFunction");
		m_grad = SGVector<float64_t>();
		SG_ADD(
			&m_grad, "GradientModelSelectionCostFunction__m_grad",
			"grad in GradientModelSelectionCostFunction");
	}

	std::shared_ptr<GradientModelSelection >m_obj;
	void* m_func_data;
	SGVector<float64_t> m_val;
	SGVector<float64_t> m_grad;
};


/** structure used for NLopt callback function */
struct nlopt_params
{
	/** pointer to current combination */
	std::shared_ptr<ParameterCombination> current_combination;

	/** pointer to parmeter dictionary */
	std::shared_ptr<CMap<TParameter*, SGObject*>> parameter_dictionary;

	/** do we want to print the state? */
	bool print_state;
};

float64_t GradientModelSelection::get_cost(SGVector<float64_t> model_vars, SGVector<float64_t> model_grads, void* func_data)
{
	REQUIRE(func_data!=NULL, "func_data must set\n");
	REQUIRE(model_vars.vlen==model_grads.vlen, "length of variable (%d) and gradient (%d) must equal\n",
		model_vars.vlen, model_grads.vlen);

	nlopt_params* params=(nlopt_params*)func_data;

	auto current_combination=params->current_combination;
	auto parameter_dictionary=params->parameter_dictionary;
	bool print_state=params->print_state;

	index_t offset=0;

	// set parameters from vector model_vars
	for (auto i : SG_PROGRESS(range(parameter_dictionary->get_num_elements())))
	{
		CMapNode<TParameter*, SGObject*>* node=parameter_dictionary->get_node_ptr(i);

		TParameter* param=node->key;
		auto parent=node->data;

		if (param->m_datatype.m_ctype==CT_VECTOR ||
				param->m_datatype.m_ctype==CT_SGVECTOR ||
				param->m_datatype.m_ctype==CT_SGMATRIX ||
				param->m_datatype.m_ctype==CT_MATRIX)
		{

			for (index_t j=0; j<param->m_datatype.get_num_elements(); j++)
			{

				bool result=current_combination->set_parameter(param->m_name,
						model_vars[offset++],	parent, j);
				 REQUIRE(result, "Parameter %s not found in combination tree\n",
						 param->m_name)
			}
		}
		else
		{
			bool result=current_combination->set_parameter(param->m_name,
					model_vars[offset++], parent);
			REQUIRE(result, "Parameter %s not found in combination tree\n",
					param->m_name)
		}
	}

	// apply current combination to the machine
	auto machine=m_machine_eval->get_machine();
	current_combination->apply_to_machine(machine);
	if (print_state)
	{
		SG_SPRINT("Current combination\n");
		current_combination->print_tree();
	}

	// evaluate the machine
	auto evaluation_result=m_machine_eval->evaluate();
	auto gradient_result = evaluation_result->as<GradientResult>();

	if (print_state)
	{
		SG_SPRINT("Current result\n");
		gradient_result->print_result();
	}

	// get value of the function, gradients and parameter dictionary
	SGVector<float64_t> value=gradient_result->get_value();

	float64_t cost = SGVector<float64_t>::sum(value);

	if (Math::is_nan(cost) || std::isinf(cost))
	{
		if (m_machine_eval->get_evaluation_direction()==ED_MINIMIZE)
			return cost;
		else
			return -cost;
	}

	auto gradient=gradient_result->get_gradient();
	auto gradient_dictionary=gradient_result->get_paramter_dictionary();


	offset=0;

	// set derivative for each parameter from parameter dictionary
	for (index_t i=0; i<parameter_dictionary->get_num_elements(); i++)
	{
		CMapNode<TParameter*, SGObject*>* node=parameter_dictionary->get_node_ptr(i);

		SGVector<float64_t> derivative;

		for (index_t j=0; j<gradient_dictionary->get_num_elements(); j++)
		{
			CMapNode<TParameter*, SGObject*>* gradient_node=
				gradient_dictionary->get_node_ptr(j);

			if (gradient_node->data==node->data &&
					!strcmp(gradient_node->key->m_name, node->key->m_name))
			{
				derivative=gradient->get_element(gradient_node->key);
			}
		}

		REQUIRE(derivative.vlen, "Can't find gradient wrt %s parameter!\n",
				node->key->m_name);

		sg_memcpy(model_grads.vector+offset, derivative.vector, sizeof(float64_t)*derivative.vlen);

		offset+=derivative.vlen;
	}

	if (m_machine_eval->get_evaluation_direction()==ED_MINIMIZE)
	{
		return cost;
	}
	else
	{
		model_grads.scale(-1);
		return -cost;
	}

}
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

void GradientModelSelection::set_minimizer(std::shared_ptr<FirstOrderMinimizer> minimizer)
{
	REQUIRE(minimizer!=NULL, "Minimizer must set\n");


	m_mode_minimizer=minimizer;
}


GradientModelSelection::GradientModelSelection() : ModelSelection()
{
	init();
}

GradientModelSelection::GradientModelSelection(std::shared_ptr<MachineEvaluation> machine_eval,
		std::shared_ptr<ModelSelectionParameters> model_parameters)
		: ModelSelection(machine_eval, model_parameters)
{
	init();
}

GradientModelSelection::~GradientModelSelection()
{

}

void GradientModelSelection::init()
{
	m_mode_minimizer = std::make_shared<LBFGSMinimizer>();


	SG_ADD((std::shared_ptr<SGObject>*)&m_mode_minimizer,
		"mode_minimizer", "Minimizer used in mode selection");

}

std::shared_ptr<ParameterCombination> GradientModelSelection::select_model(bool print_state)
{
	if (!m_model_parameters)
	{
		auto machine=m_machine_eval->get_machine();

		auto current_combination=std::make_shared<ParameterCombination>(machine);


		if (print_state)
		{
			SG_PRINT("Initial combination:\n");
			current_combination->print_tree();
		}

		// get total length of variables
		index_t total_variables=current_combination->get_parameters_length();

		// build parameter->value map
		auto argument=
			std::make_shared<CMap<TParameter*, SGVector<float64_t>>>();
		current_combination->build_parameter_values_map(argument);

		//  unroll current parameter combination into vector
		SGVector<float64_t> model_vars = SGVector<float64_t>(total_variables);

		index_t offset=0;

		for (index_t i=0; i<argument->get_num_elements(); i++)
		{
			CMapNode<TParameter*, SGVector<float64_t> >* node=argument->get_node_ptr(i);
			sg_memcpy(model_vars.vector+offset, node->data.vector, sizeof(float64_t)*node->data.vlen);
			offset+=node->data.vlen;
		}



		// build parameter->sgobject map from current parameter combination
		auto parameter_dictionary=
			std::make_shared<CMap<TParameter*, SGObject*>>();
		current_combination->build_parameter_parent_map(parameter_dictionary);

		//data for computing the gradient
		nlopt_params params;

		params.current_combination=current_combination;
		params.print_state=print_state;
		params.parameter_dictionary=parameter_dictionary;

		// choose evaluation direction (minimize or maximize objective function)
		if (print_state)
		{
			if (m_machine_eval->get_evaluation_direction()==ED_MINIMIZE)
			{
				SG_PRINT("Minimizing objective function:\n");
			}
			else
			{
				SG_PRINT("Maximizing objective function:\n");
			}
		}

		auto cost_fun=std::make_shared<GradientModelSelectionCostFunction>();
		cost_fun->set_target(shared_from_this()->as<GradientModelSelection>());
		cost_fun->set_variables(model_vars);
		cost_fun->set_func_data(&params);

		m_mode_minimizer->set_cost_function(cost_fun);
		m_mode_minimizer->minimize();
		m_mode_minimizer->unset_cost_function(false);


		if (print_state)
		{
			SG_PRINT("Best combination:\n");
			current_combination->print_tree();
		}


		return current_combination;
	}
	else
	{
		SG_NOTIMPLEMENTED
		return NULL;
	}
}

}
