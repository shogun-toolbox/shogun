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
	int32_t n_subtasks = m_tasks->get_num_elements();
	SGVector<index_t> ind(n_subtasks+1);
	index_t* min_idxs = SG_MALLOC(index_t, n_subtasks);
	index_t* max_idxs = SG_MALLOC(index_t, n_subtasks);
	index_t* task_idxs_min = SG_MALLOC(index_t, n_subtasks);
	index_t* task_idxs_max = SG_MALLOC(index_t, n_subtasks);
	CTask* iter_task = (CTask*)(m_tasks->get_first_element());
	for (int32_t i=0; i<n_subtasks; i++)
	{
		min_idxs[i] = iter_task->get_min_index(); 
		max_idxs[i] = iter_task->get_max_index();
		task_idxs_min[i] = i;
		task_idxs_max[i] = i;
		SG_UNREF(iter_task);
		iter_task = (CTask*)(m_tasks->get_next_element());
	}
	CMath::qsort_index(min_idxs, task_idxs_min, n_subtasks);
	CMath::qsort_index(max_idxs, task_idxs_max, n_subtasks);
	
	for (int32_t i=0; i<n_subtasks; i++)
	{
		if (task_idxs_min[i] != task_idxs_max[i])
			SG_ERROR("Tasks do overlap and it is not supported, please use TaskOverlappingGroup\n");
	}
	if (min_idxs[0] != 0) 
		SG_ERROR("Task with smallest indices start from %d while 0 is required\n", min_idxs[0]);
	else
		ind[0] = 0;
	for (int32_t i=1; i<n_subtasks; i++)
	{
		if (min_idxs[i] > max_idxs[i-1])
			SG_ERROR("There is an unsupported gap between %d and %d vectors\n", max_idxs[i-1], min_idxs[i]);
		else if (min_idxs[i] < max_idxs[i-1]) 
			SG_ERROR("Tasks do overlap and it is not supported, please use TaskOverlappingGroup\n");

		ind[i] = min_idxs[i];
	}
	ind[n_subtasks] = max_idxs[n_subtasks-1];

#ifdef DEBUG_SLEP
	ind.display_vector();
#endif

	SG_FREE(min_idxs);
	SG_FREE(max_idxs);
	SG_FREE(task_idxs_min);
	SG_FREE(task_idxs_max);
	return ind;
}
