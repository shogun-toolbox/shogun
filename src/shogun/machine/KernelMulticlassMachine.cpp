/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#include <set>
#include <map>

#include <shogun/machine/KernelMulticlassMachine.h>

using namespace shogun;

void CKernelMulticlassMachine::store_model_features()
{
    CKernel *kernel = ((CKernelMachine *)m_machine)->get_kernel();
	if (!kernel)
		SG_ERROR("kernel is needed to store SV features.\n");

	CFeatures* lhs = kernel->get_lhs();
	if (!lhs)
		SG_ERROR("kernel lhs is needed to store SV features.\n");

    std::set<int32_t> all_sv;

    for (int32_t i=0; i < m_machines->get_num_elements(); ++i)
    {
        CKernelMachine *machine = (CKernelMachine *)get_machine(i);
        for (int32_t j=0; j < machine->get_num_support_vectors(); ++j)
            all_sv.insert(machine->get_support_vector(j));

        SG_UNREF(machine);
    }

    int32_t i=0;
    SGVector<int32_t> sv_idx(all_sv.size());
    for (std::set<int32_t>::iterator it = all_sv.begin(); it != all_sv.end(); ++it)
        sv_idx[i++] = *it;

    std::map<int32_t, int32_t> sv_ridx;
    for (i=0; i < sv_idx.vlen; ++i)
        sv_ridx.insert(std::make_pair(sv_idx[i], i));

	CFeatures* sv_features=lhs->copy_subset(sv_idx);

    for (i=0; i < m_machines->get_num_elements(); ++i)
    {
        CKernelMachine *machine = (CKernelMachine *)get_machine(i);
        CKernel *sub_kernel = machine->get_kernel();
        CFeatures *sub_rhs = sub_kernel->get_rhs();
        sub_kernel->init(sv_features, sub_rhs);

        for (int32_t j=0; j < machine->get_num_support_vectors(); ++j)
            machine->set_support_vector(j, sv_ridx[machine->get_support_vector(j)]);

        SG_UNREF(sub_rhs);
        SG_UNREF(sub_kernel);
        SG_UNREF(machine);
    }

    SG_UNREF(lhs);
    SG_UNREF(kernel);
}
