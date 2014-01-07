/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#include <machine/BaseMulticlassMachine.h>

using namespace shogun;

CBaseMulticlassMachine::CBaseMulticlassMachine()
{
	m_machines = new CDynamicObjectArray();

	SG_ADD((CSGObject**)&m_machines, "machines", "Machines that jointly make up the multi-class machine.", MS_NOT_AVAILABLE);
}

CBaseMulticlassMachine::~CBaseMulticlassMachine()
{
	SG_UNREF(m_machines);
}

int32_t CBaseMulticlassMachine::get_num_machines() const
{
	return m_machines->get_num_elements();
}

EProblemType CBaseMulticlassMachine::get_machine_problem_type() const
{
	return PT_MULTICLASS;
}

bool CBaseMulticlassMachine::is_label_valid(CLabels *lab) const
{
	return lab->get_label_type() == LT_MULTICLASS;
}

