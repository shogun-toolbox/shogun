/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#include <shogun/lib/Map.h>
#include <shogun/machine/KernelMulticlassMachine.h>

using namespace shogun;

void CKernelMulticlassMachine::store_model_features()
{
    CKernel *kernel = ((CKernelMachine *)m_machine)->get_kernel();
	if (!kernel)
		SG_ERROR("kernel is needed to store SV features.\n");

	CFeatures* lhs = kernel->get_lhs();
	CFeatures* rhs = kernel->get_rhs();
	if (!lhs)
		SG_ERROR("kernel lhs is needed to store SV features.\n");

    CMap<int32_t, int32_t> all_sv;
    for (int32_t i=0; i < m_machines->get_num_elements(); ++i)
    {
        CKernelMachine *machine = (CKernelMachine *)get_machine(i);
        for (int32_t j=0; j < machine->get_num_support_vectors(); ++j)
            all_sv.add(machine->get_support_vector(j), 0);

        SG_UNREF(machine);
    }

    SGVector<int32_t> sv_idx(all_sv.get_num_elements());
    for (int32_t i=0; i < sv_idx.vlen; ++i)
        sv_idx[i] = all_sv.get_element(i);

    for (int32_t i=0; i < sv_idx.vlen; ++i)
        *all_sv.get_element_ptr(all_sv.index_of(sv_idx[i])) = i;

	CFeatures* sv_features=lhs->copy_subset(sv_idx);

    kernel->init(sv_features, rhs);

    for (int32_t i=0; i < m_machines->get_num_elements(); ++i)
    {
        CKernelMachine *machine = (CKernelMachine *)get_machine(i);

        for (int32_t j=0; j < machine->get_num_support_vectors(); ++j)
            machine->set_support_vector(j, all_sv.get_element(all_sv.index_of(machine->get_support_vector(j))));

        SG_UNREF(machine);
    }

    SG_UNREF(lhs);
    SG_UNREF(rhs);
    SG_UNREF(kernel);
}
