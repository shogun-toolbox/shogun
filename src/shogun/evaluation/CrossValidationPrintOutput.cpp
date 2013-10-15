/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Sergey Lisitsyn
 * Written (W) 2012 Heiko Strathmann
 */

#include <shogun/evaluation/CrossValidationPrintOutput.h>
#include <shogun/machine/LinearMachine.h>
#include <shogun/machine/LinearMulticlassMachine.h>
#include <shogun/machine/KernelMachine.h>
#include <shogun/machine/KernelMulticlassMachine.h>
#include <shogun/kernel/CombinedKernel.h>
#include <shogun/classifier/mkl/MKL.h>
#include <shogun/classifier/mkl/MKLMulticlass.h>

using namespace shogun;

void CCrossValidationPrintOutput::init_num_runs(index_t num_runs,
		const char* prefix)
{
	SG_PRINT("%scross validation number of runs %d\n", prefix, num_runs)
}

/** init number of folds */
void CCrossValidationPrintOutput::init_num_folds(index_t num_folds,
		const char* prefix)
{
	SG_PRINT("%scross validation number of folds %d\n", prefix, num_folds)
}

void CCrossValidationPrintOutput::update_run_index(index_t run_index,
		const char* prefix)
{
	SG_PRINT("%scross validation run %d\n", prefix, run_index)
}

void CCrossValidationPrintOutput::update_fold_index(index_t fold_index,
		const char* prefix)
{
	SG_PRINT("%sfold %d\n", prefix, fold_index)
}

void CCrossValidationPrintOutput::update_train_indices(
		SGVector<index_t> indices, const char* prefix)
{
	indices.display_vector("train_indices", prefix);
}

void CCrossValidationPrintOutput::update_test_indices(
		SGVector<index_t> indices, const char* prefix)
{
	indices.display_vector("test_indices", prefix);
}

void CCrossValidationPrintOutput::update_trained_machine(
		CMachine* machine, const char* prefix)
{
	if (dynamic_cast<CLinearMachine*>(machine))
	{
		CLinearMachine* linear_machine=(CLinearMachine*)machine;
		linear_machine->get_w().display_vector("learned_w", prefix);
		SG_PRINT("%slearned_bias=%f\n", prefix, linear_machine->get_bias())
	}

	if (dynamic_cast<CKernelMachine*>(machine))
	{
		CKernelMachine* kernel_machine=(CKernelMachine*)machine;
		kernel_machine->get_alphas().display_vector("learned_alphas", prefix);
		SG_PRINT("%slearned_bias=%f\n", prefix, kernel_machine->get_bias())
	}

	if (dynamic_cast<CLinearMulticlassMachine*>(machine)
			|| dynamic_cast<CKernelMulticlassMachine*>(machine))
	{
		/* append one tab to prefix */
		char* new_prefix=append_tab_to_string(prefix);

		CMulticlassMachine* mc_machine=(CMulticlassMachine*)machine;
		for (int i=0; i<mc_machine->get_num_machines(); i++)
		{
			CMachine* sub_machine=mc_machine->get_machine(i);
            //SG_PRINT("%smulti-class machine %d:\n", i, sub_machine)
			this->update_trained_machine(sub_machine, new_prefix);
			SG_UNREF(sub_machine);
		}

		/* clean up */
		SG_FREE(new_prefix);
	}

	if (dynamic_cast<CMKL*>(machine))
	{
		CMKL* mkl=(CMKL*)machine;
		CCombinedKernel* kernel=dynamic_cast<CCombinedKernel*>(
				mkl->get_kernel());
		kernel->get_subkernel_weights().display_vector("MKL sub-kernel weights",
				prefix);
		SG_UNREF(kernel);
	}

	if (dynamic_cast<CMKLMulticlass*>(machine))
	{
		CMKLMulticlass* mkl=(CMKLMulticlass*)machine;
		CCombinedKernel* kernel=dynamic_cast<CCombinedKernel*>(
				mkl->get_kernel());
		kernel->get_subkernel_weights().display_vector("MKL sub-kernel weights",
				prefix);
		SG_UNREF(kernel);
	}
}

void CCrossValidationPrintOutput::update_test_result(CLabels* results,
		const char* prefix)
{
	results->get_values().display_vector("test_labels", prefix);
}

void CCrossValidationPrintOutput::update_test_true_result(CLabels* results,
		const char* prefix)
{
	results->get_values().display_vector("true_labels", prefix);
}

void CCrossValidationPrintOutput::update_evaluation_result(float64_t result,
		const char* prefix)
{
	SG_PRINT("%sevaluation result=%f\n", prefix, result)
}

char* CCrossValidationPrintOutput::append_tab_to_string(const char* string)
{
	/* allocate memory, concatenate and add termination character */
	index_t len=strlen(string);
	char* new_prefix=SG_MALLOC(char, len+2);
	memcpy(new_prefix, string, sizeof(char*)*len);
	new_prefix[len]='\t';
	new_prefix[len+1]='\0';

	return new_prefix;
}
