/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#include <shogun/transfer/multitask/TaskTree.h>

using namespace shogun;

CTaskTree::CTaskTree() : CTaskRelation(), m_root_task(NULL)
{

}

CTaskTree::CTaskTree(CTask* root_task) : CTaskRelation(),
	m_root_task(NULL)
{
	set_root_task(root_task);
}

CTaskTree::~CTaskTree()
{
	SG_UNREF(m_root_task);
}

CTask* CTaskTree::get_root_task() const
{
	return m_root_task;
}

void CTaskTree::set_root_task(CTask* root_task)
{
	SG_REF(root_task);
	SG_UNREF(m_root_task);
	m_root_task = root_task;
}

SGVector<index_t> CTaskTree::get_SLEP_ind()
{
	CList* tasks = new CList(true);
	collect_leaf_tasks_recursive(m_root_task, tasks);
	check_task_list(tasks);

	SG_DEBUG("Collected %d leaf tasks\n", tasks->get_num_elements());

	SGVector<index_t> ind(tasks->get_num_elements()+1);

	int t_i = 0;
	ind[0] = 0;
	CTask* iterator = (CTask*)tasks->get_first_element();
	do
	{
		ind[t_i+1] = iterator->get_max_index();
		SG_DEBUG("Task = [%d,%d]\n", iterator->get_min_index(), iterator->get_max_index());
		SG_UNREF(iterator);
		t_i++;
	} 
	while ((iterator = (CTask*)tasks->get_next_element()) != NULL);

	SG_UNREF(tasks);

	return ind;
}

SGVector<float64_t> CTaskTree::get_SLEP_ind_t()
{
	CList* tasks = new CList(true);
	collect_tasks_recursive(m_root_task, tasks);

	SG_DEBUG("Collected %d tree tasks\n", tasks->get_num_elements());

	SGVector<float64_t> ind_t(3*tasks->get_num_elements());

	int t_i = 0;
	CTask* iterator = (CTask*)tasks->get_first_element();
	do
	{
		ind_t[t_i*3] = iterator->get_min_index();
		ind_t[t_i*3+1] = iterator->get_max_index();
		ind_t[t_i*3+2] = iterator->get_weight();
		SG_DEBUG("Task = [%f,%f,%f]\n", ind_t[t_i*3], ind_t[t_i*3+1], ind_t[t_i*3+2]);
		SG_UNREF(iterator);
		t_i++;
	} 
	while ((iterator = (CTask*)tasks->get_next_element()) != NULL);

	SG_UNREF(tasks);

	return ind_t;
}

bool CTaskTree::is_valid() const
{
	return true;
}

void CTaskTree::collect_tasks_recursive(CTask* subtree_root_task, CList* list)
{
	list->append_element(subtree_root_task);
	
	CList* subtasks = subtree_root_task->get_subtasks();
	if (subtasks->get_num_elements() != 0)
	{
		CTask* iterator = (CTask*)subtasks->get_first_element();
		do
		{
			collect_tasks_recursive(iterator, list);
			SG_UNREF(iterator);
		} 
		while ((iterator = (CTask*)subtasks->get_next_element()) != NULL);
	}
}

void CTaskTree::collect_leaf_tasks_recursive(CTask* subtree_root_task, CList* list)
{
	CList* subtasks = subtree_root_task->get_subtasks();
	if (subtasks->get_num_elements() == 0)
	{
		list->append_element(subtree_root_task);
	}
	else
	{
		CTask* iterator = (CTask*)subtasks->get_first_element();
		do
		{
			collect_leaf_tasks_recursive(iterator, list);
			SG_UNREF(iterator);
		} 
		while ((iterator = (CTask*)subtasks->get_next_element()) != NULL);
	}
	SG_UNREF(subtasks);
}
