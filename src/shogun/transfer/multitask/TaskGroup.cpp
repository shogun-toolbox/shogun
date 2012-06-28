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
	m_tasks = new CList(true);
}

CTaskGroup::~CTaskGroup()
{
	SG_UNREF(m_tasks);
}

void CTaskGroup::add_task(CTask* task)
{
	m_tasks->push(task);
}

void CTaskGroup::remove_task(CTask* task)
{
	SG_NOTIMPLEMENTED;
}

bool CTaskGroup::is_valid() const
{
	return true;
}

SGVector<index_t> CTaskGroup::get_SLEP_ind()
{
	check_task_list(m_tasks);
	int32_t n_subtasks = m_tasks->get_num_elements();
	SG_PRINT("Number of subtasks = %d\n", n_subtasks);
	SGVector<index_t> ind(n_subtasks+1);

	CTask* iterator = (CTask*)(m_tasks->get_first_element());
	ind[0] = 0;
	int32_t i = 0;
	do
	{
		ind[i+1] = iterator->get_max_index();
		SG_UNREF(iterator);
		i++;
	}
	while ((iterator = (CTask*)m_tasks->get_next_element()) != NULL);

	ind.display_vector();

	return ind;
}
