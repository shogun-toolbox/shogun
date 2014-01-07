/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Written (W) 2012 Heiko Strathmann
 * Copyright (C) 2012 Chiyuan Zhang
 */

#include <lib/Set.h>
#include <machine/KernelMulticlassMachine.h>

using namespace shogun;

void CKernelMulticlassMachine::store_model_features()
{
	CKernel *kernel= m_kernel;
	if (!kernel)
		SG_ERROR("%s::store_model_features(): kernel is needed to store SV "
				"features.\n", get_name());

	CFeatures* lhs=kernel->get_lhs();
	CFeatures* rhs=kernel->get_rhs();
	if (!lhs)
	{
		SG_ERROR("%s::store_model_features(): kernel lhs is needed to store "
		"SV features.\n", get_name());
	}

	/* this map will be abused as a map */
	CSet<index_t> all_sv;
	for (index_t i=0; i<m_machines->get_num_elements(); ++i)
	{
		CKernelMachine *machine=(CKernelMachine *)get_machine(i);
		for (index_t j=0; j<machine->get_num_support_vectors(); ++j)
			all_sv.add(machine->get_support_vector(j));

		SG_UNREF(machine);
	}

	/* convert map to vector of SV */
	SGVector<index_t> sv_idx(all_sv.get_num_elements());
	for (index_t i=0; i<sv_idx.vlen; ++i)
		sv_idx[i]=*all_sv.get_element_ptr(i);

	CFeatures* sv_features=lhs->copy_subset(sv_idx);

	/* now, features are replaced by concatenated SV features */
	kernel->init(sv_features, rhs);

	/* was SG_REF'ed by copy_subset */
	SG_UNREF(sv_features);

	/* now the old SV indices have to be mapped to the new features */

	/* update SV of all machines */
	for (int32_t i=0; i<m_machines->get_num_elements(); ++i)
	{
		CKernelMachine *machine=(CKernelMachine *)get_machine(i);

		/* for each machine, replace SV by index in sv_idx array */
		for (int32_t j=0; j<machine->get_num_support_vectors(); ++j)
		{
			/* get index of SV in old features */
			index_t current_sv_idx=machine->get_support_vector(j);

			/* the position of this old index in the map is the position of
			 * the SV in the new features */
			index_t new_sv_idx=all_sv.index_of(current_sv_idx);

			machine->set_support_vector(j, new_sv_idx);
		}

		SG_UNREF(machine);
	}

	SG_UNREF(lhs);
	SG_UNREF(rhs);
}

CKernelMulticlassMachine::CKernelMulticlassMachine() : CMulticlassMachine(), m_kernel(NULL)
{
	SG_ADD((CSGObject**)&m_kernel,"kernel", "The kernel to be used", MS_AVAILABLE);
}

/** standard constructor
 * @param strategy multiclass strategy
 * @param kernel kernel
 * @param machine kernel machine
 * @param labs labels
 */
CKernelMulticlassMachine::CKernelMulticlassMachine(CMulticlassStrategy *strategy, CKernel* kernel, CKernelMachine* machine, CLabels* labs) :
	CMulticlassMachine(strategy,(CMachine*)machine,labs), m_kernel(NULL)
{
	set_kernel(kernel);
	SG_ADD((CSGObject**)&m_kernel,"kernel", "The kernel to be used", MS_AVAILABLE);
}

/** destructor */
CKernelMulticlassMachine::~CKernelMulticlassMachine()
{
	SG_UNREF(m_kernel);
}

/** set kernel
 *
 * @param k kernel
 */
void CKernelMulticlassMachine::set_kernel(CKernel* k)
{
	((CKernelMachine*)m_machine)->set_kernel(k);
	SG_REF(k);
	SG_UNREF(m_kernel);
	m_kernel=k;
}

CKernel* CKernelMulticlassMachine::get_kernel()
{
	SG_REF(m_kernel);
	return m_kernel;
}

bool CKernelMulticlassMachine::init_machine_for_train(CFeatures* data)
{
	if (data)
		m_kernel->init(data,data);

	((CKernelMachine*)m_machine)->set_kernel(m_kernel);

	return true;
}

bool CKernelMulticlassMachine::init_machines_for_apply(CFeatures* data)
{
	if (data)
	{
		/* set data to rhs for this kernel */
		CFeatures* lhs=m_kernel->get_lhs();
		m_kernel->init(lhs, data);
		SG_UNREF(lhs);
	}

	/* set kernel to all sub-machines */
	for (int32_t i=0; i<m_machines->get_num_elements(); i++)
	{
		CKernelMachine *machine=
				(CKernelMachine*)m_machines->get_element(i);
		machine->set_kernel(m_kernel);
		SG_UNREF(machine);
	}

	return true;
}

bool CKernelMulticlassMachine::is_ready()
{
	if (m_kernel && m_kernel->get_num_vec_lhs() && m_kernel->get_num_vec_rhs())
			return true;

	return false;
}

CMachine* CKernelMulticlassMachine::get_machine_from_trained(CMachine* machine)
{
	return new CKernelMachine((CKernelMachine*)machine);
}

int32_t CKernelMulticlassMachine::get_num_rhs_vectors()
{
	return m_kernel->get_num_vec_rhs();
}

void CKernelMulticlassMachine::add_machine_subset(SGVector<index_t> subset)
{
	SG_NOTIMPLEMENTED
}

void CKernelMulticlassMachine::remove_machine_subset()
{
	SG_NOTIMPLEMENTED
}


