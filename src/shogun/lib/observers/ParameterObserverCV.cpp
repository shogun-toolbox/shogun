/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Giovanni De Toni
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

ParameterObserverCV::ParameterObserverCV() : ParameterObserver()
{
}

ParameterObserverCV::ParameterObserverCV(
    std::vector<std::string>& parameters,
    std::vector<ParameterProperties>& properties)
    : ParameterObserver(parameters, properties)
{
}

ParameterObserverCV::ParameterObserverCV(
    const std::string& filename, std::vector<std::string>& parameters,
    std::vector<ParameterProperties>& properties)
    : ParameterObserver(filename, parameters, properties)
{
}

ParameterObserverCV::ParameterObserverCV(std::vector<std::string>& parameters)
    : ParameterObserver(parameters)
{
}

ParameterObserverCV::ParameterObserverCV(
    std::vector<ParameterProperties>& properties)
    : ParameterObserver(properties)
{
}

ParameterObserverCV::~ParameterObserverCV()
{
}

void ParameterObserverCV::on_next_impl(const shogun::TimedObservedValue& value)
{
	try
	{
		auto recalled_value =
		        value.first->get(value.first->get<std::string>("name"))->as<CrossValidationStorage>();

		/* Print information on screen if enabled*/
		print_observed_value(recalled_value);
	}
	catch (ShogunException& e)
	{
		io::print(
		    "{}: Received an observed value named {} which is a not a "
		    "CrossValidationStorage object"
		    ", therefore it was ignored.",
		    this->get_name(), value.first->get<std::string>("name").c_str());
	}
}

void ParameterObserverCV::on_error(std::exception_ptr ptr)
{
}

void ParameterObserverCV::on_complete()
{
}

void ParameterObserverCV::print_observed_value(
    const std::shared_ptr<CrossValidationStorage>& value) const
{
	for (index_t i = 0; i < value->get<index_t>("num_folds"); i++)
	{
		auto f = value->get("folds", i);
		io::print("\n");
		io::print(
		    "Current run index: {}\n", f->get<index_t>("current_run_index"));
		io::print(
		    "Current fold index: {}\n", f->get<index_t>("current_fold_index"));
		f->get<SGVector<index_t>>("train_indices")
		    .display_vector("Train Indices ");
		f->get<SGVector<index_t>>("test_indices")
		    .display_vector("Test Indices ");
		print_machine_information(f->get<Machine>("trained_machine"));
		f->get<Labels>("test_result")
		    ->get_values()
		    .display_vector("Test Labels ");
		f->get<Labels>("test_true_result")
		    ->get_values()
		    .display_vector("Test True Label ");
		io::print(
		    "Evaluation result: {}\n", f->get<float64_t>("evaluation_result"));
	}
}

void ParameterObserverCV::print_machine_information(const std::shared_ptr<Machine>& machine) const
{
	if (std::dynamic_pointer_cast<LinearMachine>(machine))
	{
		auto linear_machine = std::static_pointer_cast<LinearMachine>(machine);
		linear_machine->get_w().display_vector("Learned Weights = ");
		io::print("Learned Bias = {}\n", linear_machine->get_bias());
	}

	if (std::dynamic_pointer_cast<KernelMachine>(machine))
	{
		auto kernel_machine = machine->as<KernelMachine>();
		kernel_machine->get_alphas().display_vector("Learned alphas = ");
		io::print("Learned Bias = {}\n", kernel_machine->get_bias());
	}

	if (std::dynamic_pointer_cast<LinearMulticlassMachine>(machine) ||
	    std::dynamic_pointer_cast<KernelMulticlassMachine>(machine))
	{
		auto mc_machine = machine->as<MulticlassMachine>();
		for (int i = 0; i < mc_machine->get_num_machines(); i++)
		{
			auto sub_machine = mc_machine->get_machine(i);
			this->print_machine_information(sub_machine);
		}
	}

	if (std::dynamic_pointer_cast<MKL>(machine))
	{
		auto mkl = machine->as<MKL>();
		auto kernel = mkl->get_kernel()->as<CombinedKernel>();
		kernel->get_subkernel_weights().display_vector(
		    "MKL sub-kernel weights =");

	}

	if (std::dynamic_pointer_cast<MKLMulticlass>(machine))
	{
		auto mkl = machine->as<MKLMulticlass>();
		auto kernel = mkl->get_kernel()->as<CombinedKernel>();
		kernel->get_subkernel_weights().display_vector(
		    "MKL sub-kernel weights =");

	}
}

