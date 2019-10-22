/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Roman Votyakov, Soeren Sonnenburg, Jacob Walker, Heiko Strathmann, 
 *          Sergey Lisitsyn
 */

#include <shogun/modelselection/ModelSelection.h>

#ifdef ENABLE_MODEL_SELECTION
#include <shogun/modelselection/ModelSelectionParameters.h>
#include <shogun/evaluation/CrossValidation.h>
#include <shogun/base/Parameter.h>

using namespace shogun;

ModelSelection::ModelSelection()
{
	init();
}

ModelSelection::ModelSelection(std::shared_ptr<MachineEvaluation> machine_eval,
	std::shared_ptr<ModelSelectionParameters> model_parameters)
{
	init();

	m_model_parameters=model_parameters;
	

	m_machine_eval=machine_eval;
	
}

void ModelSelection::init()
{
	m_model_parameters=NULL;
	m_machine_eval=NULL;

	SG_ADD((std::shared_ptr<SGObject>*)&m_model_parameters, "model_parameters",
			"Parameter tree for model selection");

	SG_ADD((std::shared_ptr<SGObject>*)&m_machine_eval, "machine_evaluation",
			"Machine evaluation strategy");
}

ModelSelection::~ModelSelection()
{
	
	
}

#endif

