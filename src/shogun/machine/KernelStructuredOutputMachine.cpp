/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#include <machine/KernelStructuredOutputMachine.h>

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
