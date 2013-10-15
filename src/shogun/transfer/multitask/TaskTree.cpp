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

int32_t count_leaf_tasks_recursive(CTask* subtree_root_block)
{
	CList* sub_tasks = subtree_root_block->get_subtasks();
	int32_t n_sub_tasks = sub_tasks->get_num_elements();
	if (n_sub_tasks==0)
	{
		SG_UNREF(sub_tasks);
		return 1;
	}
	else
	{
		int32_t sum = 0;
		CTask* iterator = (CTask*)sub_tasks->get_first_element();
		do
		{
			sum += count_leaf_tasks_recursive(iterator);
			SG_UNREF(iterator);
		}
		while ((iterator = (CTask*)sub_tasks->get_next_element()) != NULL);

		SG_UNREF(sub_tasks);
		return sum;
	}
}

void collect_tree_tasks_recursive(CTask* subtree_root_block, vector<task_tree_node_t>* tree_nodes, int low)
{
	int32_t lower = low;
	CList* sub_blocks = subtree_root_block->get_subtasks();
	if (sub_blocks->get_num_elements()>0)
	{
		CTask* iterator = (CTask*)sub_blocks->get_first_element();
		do
		{
			if (iterator->get_num_subtasks()>0)
			{
				int32_t n_leaves = count_leaf_tasks_recursive(iterator);
				//SG_SDEBUG("Block [%d %d] has %d leaf childs \n",iterator->get_min_index(), iterator->get_max_index(), n_leaves)
				tree_nodes->push_back(task_tree_node_t(lower,lower+n_leaves-1,iterator->get_weight()));
				collect_tree_tasks_recursive(iterator, tree_nodes, lower);
				lower = lower + n_leaves;
			}
			else
				lower++;
			SG_UNREF(iterator);
		}
		while ((iterator = (CTask*)sub_blocks->get_next_element()) != NULL);
	}
	SG_UNREF(sub_blocks);
}

void collect_leaf_tasks_recursive(CTask* subtree_root_block, CList* list)
{
	CList* sub_blocks = subtree_root_block->get_subtasks();
	if (sub_blocks->get_num_elements() == 0)
	{
		list->append_element(subtree_root_block);
	}
	else
	{
		CTask* iterator = (CTask*)sub_blocks->get_first_element();
		do
		{
			collect_leaf_tasks_recursive(iterator, list);
			SG_UNREF(iterator);
		}
		while ((iterator = (CTask*)sub_blocks->get_next_element()) != NULL);
	}
	SG_UNREF(sub_blocks);
}

int32_t count_leaft_tasks_recursive(CTask* subtree_root_block)
{
	CList* sub_blocks = subtree_root_block->get_subtasks();
	int32_t r = 0;
	if (sub_blocks->get_num_elements() == 0)
	{
		return 1;
	}
	else
	{
		CTask* iterator = (CTask*)sub_blocks->get_first_element();
		do
		{
			r += count_leaf_tasks_recursive(iterator);
			SG_UNREF(iterator);
		}
		while ((iterator = (CTask*)sub_blocks->get_next_element()) != NULL);
	}
	SG_UNREF(sub_blocks);
	return r;
}

CTaskTree::CTaskTree() : CTaskRelation(),
	m_root_task(NULL)
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

SGVector<index_t>* CTaskTree::get_tasks_indices() const
{
	CList* blocks = new CList(true);
	collect_leaf_tasks_recursive(m_root_task, blocks);
	SG_DEBUG("Collected %d leaf blocks\n", blocks->get_num_elements())
	//check_blocks_list(blocks);

	//SGVector<index_t> ind(blocks->get_num_elements()+1);

	int t_i = 0;
	//ind[0] = 0;
	//
	SGVector<index_t>* tasks_indices = SG_MALLOC(SGVector<index_t>, blocks->get_num_elements());
	CTask* iterator = (CTask*)blocks->get_first_element();
	do
	{
		tasks_indices[t_i] = iterator->get_indices();
		//REQUIRE(iterator->is_contiguous(),"Task is not contiguous")
		//ind[t_i+1] = iterator->get_indices()[iterator->get_indices().vlen-1] + 1;
		//SG_DEBUG("Block = [%d,%d]\n", iterator->get_min_index(), iterator->get_max_index())
		SG_UNREF(iterator);
		t_i++;
	}
	while ((iterator = (CTask*)blocks->get_next_element()) != NULL);

	SG_UNREF(blocks);

	return tasks_indices;
}

int32_t CTaskTree::get_num_tasks() const
{
	return count_leaf_tasks_recursive(m_root_task);
}

SGVector<float64_t> CTaskTree::get_SLEP_ind_t()
{
	int n_blocks = get_num_tasks() - 1;
	SG_DEBUG("Number of blocks = %d \n", n_blocks)

	vector<task_tree_node_t> tree_nodes = vector<task_tree_node_t>();

	collect_tree_tasks_recursive(m_root_task, &tree_nodes,1);

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

	return ind_t;
}
