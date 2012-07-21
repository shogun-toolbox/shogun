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
	init();
}

CTaskGroup::~CTaskGroup()
{
	SG_UNREF(m_tasks);
}

void CTaskGroup::init()
{
	m_tasks = new CDynamicObjectArray();
}

void CTaskGroup::append_task(CTask* task)
{
	m_tasks->append_element(task);
}

int32_t CTaskGroup::get_num_tasks()
{
	return m_tasks->get_num_elements();
}

SGVector<index_t> CTaskGroup::get_SLEP_ind()
{
	// glance over tasks to check whether they are non-contiguous
	for (int32_t i=0; i<m_tasks->get_num_elements(); i++)
	{
		CTask* task = (CTask*)m_tasks->get_element(i);
		REQUIRE(task->is_contiguous(),"SLEP solver doesn't support non-contiguous tasks yet");
		SG_UNREF(task);
	}

	int32_t n_tasks = m_tasks->get_num_elements();
	SGVector<index_t> ind(n_tasks+1);
	ind[0] = 0;
	for (int32_t i=1; i<n_tasks; i++)
	{
		CTask* task_previous = (CTask*)m_tasks->get_element(i-1);
		CTask* task = (CTask*)m_tasks->get_element(i);
		REQUIRE(task->get_indices()[0]-1==task_previous->get_indices()[task_previous->get_indices().vlen-1],"There is a gap");
		ind[i] = task->get_indices()[0];
		SG_UNREF(task);
	}
	CTask* task = (CTask*)m_tasks->get_element(n_tasks-1);
	ind[ind.vlen-1] = task->get_indices()[task->get_indices().vlen-1]+1;
	SG_UNREF(task);
	return ind;
}

SGVector<index_t>* CTaskGroup::get_tasks_indices()
{
	int32_t n_tasks = m_tasks->get_num_elements();
	SG_DEBUG("Number of tasks = %d\n", n_tasks);
	
	SGVector<index_t>* tasks_indices = SG_MALLOC(SGVector<index_t>, n_tasks);
	for (int32_t i=0; i<n_tasks; i++)
	{
		new (&tasks_indices[i]) SGVector<index_t>();
		CTask* task = (CTask*)m_tasks->get_element(i);
		tasks_indices[i] = task->get_indices();
		SG_UNREF(task);
	}

	return tasks_indices;
}
