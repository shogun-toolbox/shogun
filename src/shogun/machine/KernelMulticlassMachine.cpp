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

#include <shogun/lib/Set.h>
#include <shogun/machine/KernelMulticlassMachine.h>

using namespace shogun;

void CKernelMulticlassMachine::store_model_features()
{
	CKernel *kernel=((CKernelMachine *)m_machine)->get_kernel();
	if (!kernel)
		SG_ERROR("kernel is needed to store SV features.\n");

	CFeatures* lhs=kernel->get_lhs();
	CFeatures* rhs=kernel->get_rhs();
	if (!lhs)
	{
		SG_ERROR("%s::store_model_features(): kernel lhs is needed to store "
		"SV features.\n");
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
	SG_UNREF(kernel);
}
