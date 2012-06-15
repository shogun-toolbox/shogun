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

CTaskTree::CTaskTree() : CTaskRelation()
{

}

CTaskTree::CTaskTree(CTask* root_task) : CTaskRelation()
{

}

CTaskTree::~CTaskTree()
{

}

SGVector<index_t> CTaskTree::get_SLEP_ind()
{
	CList* tasks = new CList();
	collect_tasks_recursive(m_root_task, tasks);
}

SGVector<float64_t> CTaskTree::get_SLEP_ind_t()
{
}

bool CTaskTree::is_valid() const
{
	return true;
}

void CTaskTree::collect_tasks_recursive(CTask* subtree_root_task, CList* list)
{
	list->append_element(subtree_root_task);
	
	CList* root_task_subtasks = subtree_root_task->get_subtasks();
}
