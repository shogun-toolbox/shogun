/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (W) 2012 Sergey Lisitsyn
 */

#include <shogun/modelselection/ModelSelectionOutput.h>
#include <shogun/base/Parameter.h>
#include <shogun/machine/LinearMachine.h>
#include <shogun/machine/LinearMulticlassMachine.h>
#include <shogun/machine/KernelMachine.h>
#include <shogun/machine/KernelMulticlassMachine.h>
#include <shogun/evaluation/MulticlassAccuracy.h>

using namespace shogun;

CModelSelectionOutput::CModelSelectionOutput() : CSGObject()
{
}

CModelSelectionOutput::~CModelSelectionOutput()
{
}

void CModelSelectionOutput::output_train_indices(SGVector<index_t> indices)
{
	indices.display_vector("train_indices");
}

void CModelSelectionOutput::output_test_indices(SGVector<index_t> indices)
{
	indices.display_vector("test_indices");
}

void CModelSelectionOutput::output_trained_machine(CMachine* machine)
{
	if (dynamic_cast<CLinearMachine*>(machine))
	{
		CLinearMachine* linear_machine = (CLinearMachine*)machine;
		linear_machine->get_w().display_vector("learned_w");
		SG_PRINT("learned_bias=%f\n",linear_machine->get_bias());
	}
	if (dynamic_cast<CKernelMachine*>(machine))
	{
		CKernelMachine* kernel_machine = (CKernelMachine*)machine;
		kernel_machine->get_alphas().display_vector("learned_alphas");
		SG_PRINT("learned_bias=%f\n",kernel_machine->get_bias());
	}
	if (dynamic_cast<CLinearMulticlassMachine*>(machine) ||
	    dynamic_cast<CKernelMulticlassMachine*>(machine))
	{
		CMulticlassMachine* mc_machine = (CMulticlassMachine*)machine;
		for (int i=0; i<mc_machine->get_num_machines(); i++)
		{
			CMachine* sub_machine = mc_machine->get_machine(i);
			this->output_trained_machine(sub_machine);
			SG_UNREF(sub_machine);
		}
	}
}

void CModelSelectionOutput::output_test_result(CLabels* results)
{
	results->get_confidences().display_vector("test_labels");
}

void CModelSelectionOutput::output_test_true_result(CLabels* results)
{
	results->get_confidences().display_vector("true_labels");
}

void CModelSelectionOutput::output_evaluate_result(float64_t result)
{
	SG_PRINT("evaluation result = %f\n",result);
}

void CModelSelectionOutput::add_custom_evaluation(CEvaluation* evaluation)
{
	m_custom_evaluations->append_element(evaluation);
}

void CModelSelectionOutput::output_custom_evaluations(CLabels* result, CLabels* truth)
{
	for (int32_t i=0; i<m_custom_evaluations->get_num_elements(); i++)
	{
		CEvaluation* custom_evaluation = (CEvaluation*)m_custom_evaluations->get_element(i);
		float64_t eval = custom_evaluation->evaluate(result, truth);
		SG_PRINT("%s evaluation result = %f\n",custom_evaluation->get_name(),eval);
		// dirty hack for confusion matrix
		if (dynamic_cast<CMulticlassAccuracy*>(custom_evaluation))
		{
			((CMulticlassAccuracy*)custom_evaluation)->get_confusion_matrix(result, truth).display_matrix("confusion_matrix");
		}
		SG_UNREF(custom_evaluation);
	}
}

