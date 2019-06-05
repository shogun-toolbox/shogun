/*
* BSD 3-Clause License
*
* Copyright (c) 2017, Shogun-Toolbox e.V. <shogun-team@shogun-toolbox.org>
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* * Redistributions of source code must retain the above copyright notice, this
*   list of conditions and the following disclaimer.
*
* * Redistributions in binary form must reproduce the above copyright notice,
*   this list of conditions and the following disclaimer in the documentation
*   and/or other materials provided with the distribution.
*
* * Neither the name of the copyright holder nor the names of its
*   contributors may be used to endorse or promote products derived from
*   this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
* Written (W) 2017 Giovanni De Toni
*
*/

#include <shogun/classifier/mkl/MKL.h>
#include <shogun/classifier/mkl/MKLMulticlass.h>
#include <shogun/lib/observers/ObservedValue.h>
#include <shogun/lib/observers/ParameterObserverCV.h>
#include <shogun/machine/LinearMachine.h>
#include <shogun/machine/LinearMulticlassMachine.h>
#include <shogun/util/converters.h>

using namespace shogun;

CParameterObserverCV::CParameterObserverCV(bool verbose)
    : ParameterObserver(), m_verbose(verbose)
{
}

CParameterObserverCV::~CParameterObserverCV()
{
}

void CParameterObserverCV::on_next_impl(const shogun::TimedObservedValue& value)
{
	try
	{
		CrossValidationStorage* recalled_value =
		    dynamic_cast<CrossValidationStorage*>(
		        value.first->get(value.first->get<std::string>("name")));
		SG_REF(recalled_value);

		/* Print information on screen if enabled*/
		if (m_verbose)
			print_observed_value(recalled_value);
	}
	catch (ShogunException e)
	{
		SG_PRINT(
		    "%s: Received an observed value named %s which is a not a "
		    "CrossValidationStorage object"
		    ", therefore it was ignored.",
		    this->get_name(), value.first->get<std::string>("name").c_str())
	}
}

void CParameterObserverCV::on_error(std::exception_ptr ptr)
{
}

void CParameterObserverCV::on_complete()
{
}

void CParameterObserverCV::print_observed_value(
    CrossValidationStorage* value) const
{
	for (index_t i = 0; i < value->get<index_t>("num_folds"); i++)
	{
		auto f = value->get("folds", i);
		SG_PRINT("\n")
		SG_PRINT(
		    "Current run index: %i\n", f->get<index_t>("current_run_index"))
		SG_PRINT(
		    "Current fold index: %i\n", f->get<index_t>("current_fold_index"))
		f->get<SGVector<index_t>>("train_indices")
		    .display_vector("Train Indices ");
		f->get<SGVector<index_t>>("test_indices")
		    .display_vector("Test Indices ");
		print_machine_information(f->get<CMachine*>("trained_machine"));
		f->get<CLabels*>("test_result")
		    ->get_values()
		    .display_vector("Test Labels ");
		f->get<CLabels*>("test_true_result")
		    ->get_values()
		    .display_vector("Test True Label ");
		SG_PRINT(
		    "Evaluation result: %f\n", f->get<float64_t>("evaluation_result"));
		SG_UNREF(f)
	}
}

void CParameterObserverCV::print_machine_information(CMachine* machine) const
{
	if (dynamic_cast<CLinearMachine*>(machine))
	{
		CLinearMachine* linear_machine = (CLinearMachine*)machine;
		linear_machine->get_w().display_vector("Learned Weights = ");
		SG_PRINT("Learned Bias = %f\n", linear_machine->get_bias())
	}

	if (dynamic_cast<CKernelMachine*>(machine))
	{
		CKernelMachine* kernel_machine = (CKernelMachine*)machine;
		kernel_machine->get_alphas().display_vector("Learned alphas = ");
		SG_PRINT("Learned Bias = %f\n", kernel_machine->get_bias())
	}

	if (dynamic_cast<CLinearMulticlassMachine*>(machine) ||
	    dynamic_cast<CKernelMulticlassMachine*>(machine))
	{
		CMulticlassMachine* mc_machine = (CMulticlassMachine*)machine;
		for (int i = 0; i < mc_machine->get_num_machines(); i++)
		{
			CMachine* sub_machine = mc_machine->get_machine(i);
			this->print_machine_information(sub_machine);
			SG_UNREF(sub_machine);
		}
	}

	if (dynamic_cast<CMKL*>(machine))
	{
		CMKL* mkl = (CMKL*)machine;
		CCombinedKernel* kernel =
		    dynamic_cast<CCombinedKernel*>(mkl->get_kernel());
		kernel->get_subkernel_weights().display_vector(
		    "MKL sub-kernel weights =");
		SG_UNREF(kernel);
	}

	if (dynamic_cast<CMKLMulticlass*>(machine))
	{
		CMKLMulticlass* mkl = (CMKLMulticlass*)machine;
		CCombinedKernel* kernel =
		    dynamic_cast<CCombinedKernel*>(mkl->get_kernel());
		kernel->get_subkernel_weights().display_vector(
		    "MKL sub-kernel weights =");
		SG_UNREF(kernel);
	}
}
