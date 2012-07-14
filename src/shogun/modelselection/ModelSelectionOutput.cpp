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
	if ((CLinearMachine*)machine)
	{
		CLinearMachine* linear_machine = (CLinearMachine*)machine;
		linear_machine->get_w().display_vector("learned_w");
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
