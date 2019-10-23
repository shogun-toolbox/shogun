/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Fernando Iglesias, Thoralf Klein, Shell Hu
 */

#include <shogun/machine/KernelStructuredOutputMachine.h>
#include <shogun/kernel/Kernel.h>

#include <utility>

using namespace shogun;

KernelStructuredOutputMachine::KernelStructuredOutputMachine()
: StructuredOutputMachine(), m_kernel(NULL)
{
	register_parameters();
}

KernelStructuredOutputMachine::KernelStructuredOutputMachine(
		std::shared_ptr<StructuredModel>  model,
		std::shared_ptr<StructuredLabels> labs,
		std::shared_ptr<Kernel>           kernel)
: StructuredOutputMachine(std::move(model), std::move(labs)), m_kernel(NULL)
{
	set_kernel(std::move(kernel));
	register_parameters();
}

KernelStructuredOutputMachine::~KernelStructuredOutputMachine()
{
	
}

void KernelStructuredOutputMachine::set_kernel(std::shared_ptr<Kernel> k)
{
	
	
	m_kernel = std::move(k);
}

std::shared_ptr<Kernel> KernelStructuredOutputMachine::get_kernel() const
{
	
	return m_kernel;
}

void KernelStructuredOutputMachine::register_parameters()
{
	SG_ADD((std::shared_ptr<SGObject>*)&m_kernel, "m_kernel", "The kernel", ParameterProperties::HYPER);
}
