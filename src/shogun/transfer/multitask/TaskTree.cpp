/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#include <shogun/transfer/multitask/TaskTree.h>
#include <vector>

using namespace std;
using namespace shogun;

struct task_tree_node_t
{
	task_tree_node_t(int32_t min, int32_t max, float64_t w)
	{
		t_min_index = min;
		t_max_index = max;
		weight = w;
	}
	int32_t t_min_index, t_max_index;
	float64_t weight;
};

int32_t count_leaf_tasks_recursive(CTask* subtree_root_task)
{
	CList* subtasks = subtree_root_task->get_subtasks();
	int32_t n_subtasks = subtasks->get_num_elements();
	if (n_subtasks==0)
	{
		SG_UNREF(subtasks);
		return 1;
	}
	else
	{
		int32_t sum = 0;
		CTask* iterator = (CTask*)subtasks->get_first_element();
		do
		{
			sum += count_leaf_tasks_recursive(iterator);
		}
		while ((iterator = (CTask*)subtasks->get_next_element()) != NULL);

		SG_UNREF(subtasks);
		return sum;
	}
}

void collect_tree_nodes_recursive(CTask* subtree_root_task, vector<task_tree_node_t>* tree_nodes, int low)
{
	int32_t lower = low;
	CList* subtasks = subtree_root_task->get_subtasks();
	if (subtasks->get_num_elements()>0)
	{
		CTask* iterator = (CTask*)subtasks->get_first_element();
		do
		{
			if (iterator->get_num_subtasks()>0)
			{
				int32_t n_leaves = count_leaf_tasks_recursive(iterator);
				SG_SDEBUG("Task [%d %d] has %d leaf childs \n",iterator->get_min_index(), iterator->get_max_index(), n_leaves);
				tree_nodes->push_back(task_tree_node_t(lower,lower+n_leaves-1,iterator->get_weight()));
				collect_tree_nodes_recursive(iterator, tree_nodes, lower);
				lower = lower + n_leaves;
			}
			else
				lower++;
			SG_UNREF(iterator);
		}
		while ((iterator = (CTask*)subtasks->get_next_element()) != NULL);
	}
	SG_UNREF(subtasks);
}

void collect_leaf_tasks_recursive(CTask* subtree_root_task, CList* list)
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
	SG_REF(m_root_task);
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
	SG_DEBUG("Collected %d leaf tasks\n", tasks->get_num_elements());
	check_task_list(tasks);


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
	int n_tasks = get_SLEP_ind().vlen;
	SG_DEBUG("Number of tasks = %d \n", n_tasks);

	vector<task_tree_node_t> tree_nodes = vector<task_tree_node_t>();
	
	collect_tree_nodes_recursive(m_root_task, &tree_nodes,1);

	SGVector<float64_t> ind_t(3+3*tree_nodes.size());
	// supernode
	ind_t[0] = -1;
	ind_t[1] = -1;
	ind_t[2] = 1.0;

	for (int32_t i=0; i<(int32_t)tree_nodes.size(); i++)
	{
		ind_t[3+i*3] = tree_nodes[i].t_min_index;
		ind_t[3+i*3+1] = tree_nodes[i].t_max_index;
		ind_t[3+i*3+2] = tree_nodes[i].weight;
	}

	SG_UNREF(tasks);

	return ind_t;
}

bool CTaskTree::is_valid() const
{
	return true;
}

