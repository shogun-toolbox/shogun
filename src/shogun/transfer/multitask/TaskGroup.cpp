/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#include <shogun/transfer/multitask/TaskGroup.h>

using namespace shogun;

CTaskGroup::CTaskGroup() : CTaskRelation()
{
	m_tasks = new CList();
}

CTaskGroup::~CTaskGroup()
{
	SG_UNREF(m_tasks);
}

void CTaskGroup::add_task(CTask* task)
{
	m_tasks->append_element(task);
}

bool CTaskGroup::is_valid() const
{
	return true;
}

SGVector<index_t> CTaskGroup::get_SLEP_ind()
{
	SGVector<index_t> ind(2*m_tasks->get_num_elements());
	return ind;
}
