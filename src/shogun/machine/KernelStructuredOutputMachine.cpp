/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Fernando Iglesias, Thoralf Klein, Shell Hu
 */

#include <shogun/machine/KernelStructuredOutputMachine.h>
#include <shogun/kernel/Kernel.h>

using namespace shogun;

CKernelStructuredOutputMachine::CKernelStructuredOutputMachine()
: CStructuredOutputMachine(), m_kernel(NULL)
{
	register_parameters();
}

CKernelStructuredOutputMachine::CKernelStructuredOutputMachine(
		CStructuredModel*  model,
		CStructuredLabels* labs,
		CKernel*           kernel)
: CStructuredOutputMachine(model, labs), m_kernel(NULL)
{
	set_kernel(kernel);
	register_parameters();
}

CKernelStructuredOutputMachine::~CKernelStructuredOutputMachine()
{
	SG_UNREF(m_kernel)
}

void CKernelStructuredOutputMachine::set_kernel(CKernel* k)
{
	SG_REF(k);
	SG_UNREF(m_kernel);
	m_kernel = k;
}

CKernel* CKernelStructuredOutputMachine::get_kernel() const
{
	SG_REF(m_kernel);
	return m_kernel;
}

void CKernelStructuredOutputMachine::register_parameters()
{
	SG_ADD((CSGObject**)&m_kernel, "m_kernel", "The kernel", MS_AVAILABLE);
}
