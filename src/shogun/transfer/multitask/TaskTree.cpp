/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Soeren Sonnenburg, Bjoern Esser
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

int32_t count_leaf_tasks_recursive(std::shared_ptr<Task> subtree_root_block)
{
	auto sub_tasks = subtree_root_block->get_subtasks();
	int32_t n_sub_tasks = sub_tasks->get_num_elements();
	if (n_sub_tasks==0)
	{

		return 1;
	}
	else
	{
		int32_t sum = 0;
		auto iterator = sub_tasks->get_first_element()->as<Task>();
		do
		{
			sum += count_leaf_tasks_recursive(iterator);
		}
		while ((iterator = sub_tasks->get_next_element()->as<Task>()) != NULL);


		return sum;
	}
}

void collect_tree_tasks_recursive(std::shared_ptr<Task> subtree_root_block, vector<task_tree_node_t>* tree_nodes, int low)
{
	int32_t lower = low;
	auto sub_blocks = subtree_root_block->get_subtasks();
	if (sub_blocks->get_num_elements()>0)
	{
		auto iterator = sub_blocks->get_first_element()->as<Task>();
		do
		{
			if (iterator->get_num_subtasks()>0)
			{
				int32_t n_leaves = count_leaf_tasks_recursive(iterator);
				//SG_DEBUG("Block [{} {}] has {} leaf childs ",iterator->get_min_index(), iterator->get_max_index(), n_leaves)
				tree_nodes->push_back(task_tree_node_t(lower,lower+n_leaves-1,iterator->get_weight()));
				collect_tree_tasks_recursive(iterator, tree_nodes, lower);
				lower = lower + n_leaves;
			}
			else
				lower++;
		}
		while ((iterator = sub_blocks->get_next_element()->as<Task>()) != NULL);
	}

}

void collect_leaf_tasks_recursive(std::shared_ptr<Task> subtree_root_block, std::shared_ptr<List> list)
{
	auto sub_blocks = subtree_root_block->get_subtasks();
	if (sub_blocks->get_num_elements() == 0)
	{
		list->append_element(subtree_root_block);
	}
	else
	{
		auto iterator = sub_blocks->get_first_element()->as<Task>();
		do
		{
			collect_leaf_tasks_recursive(iterator, list);
		}
		while ((iterator = sub_blocks->get_next_element()->as<Task>()) != NULL);
	}

}

int32_t count_leaft_tasks_recursive(std::shared_ptr<Task> subtree_root_block)
{
	auto sub_blocks = subtree_root_block->get_subtasks();
	int32_t r = 0;
	if (sub_blocks->get_num_elements() == 0)
	{
		return 1;
	}
	else
	{
		auto iterator = sub_blocks->get_first_element()->as<Task>();
		do
		{
			r += count_leaf_tasks_recursive(iterator);
		}
		while ((iterator = sub_blocks->get_next_element()->as<Task>()) != NULL);
	}

	return r;
}

TaskTree::TaskTree() : TaskRelation(),
	m_root_task(NULL)
{

}

TaskTree::TaskTree(std::shared_ptr<Task> root_task) : TaskRelation(),
	m_root_task(NULL)
{
	set_root_task(root_task);
}

TaskTree::~TaskTree()
{

}

SGVector<index_t>* TaskTree::get_tasks_indices() const
{
	auto blocks = std::make_shared<List>(true);
	collect_leaf_tasks_recursive(m_root_task, blocks);
	SG_DEBUG("Collected {} leaf blocks", blocks->get_num_elements())
	//check_blocks_list(blocks);

	//SGVector<index_t> ind(blocks->get_num_elements()+1);

	int t_i = 0;
	//ind[0] = 0;
	//
	SGVector<index_t>* tasks_indices = SG_MALLOC(SGVector<index_t>, blocks->get_num_elements());
	auto iterator = blocks->get_first_element()->as<Task>();
	do
	{
		tasks_indices[t_i] = iterator->get_indices();
		//require(iterator->is_contiguous(),"Task is not contiguous");
		//ind[t_i+1] = iterator->get_indices()[iterator->get_indices().vlen-1] + 1;
		//SG_DEBUG("Block = [{},{}]", iterator->get_min_index(), iterator->get_max_index())
		t_i++;
	}
	while ((iterator = blocks->get_next_element()->as<Task>()) != NULL);



	return tasks_indices;
}

int32_t TaskTree::get_num_tasks() const
{
	return count_leaf_tasks_recursive(m_root_task);
}

SGVector<float64_t> TaskTree::get_SLEP_ind_t()
{
	int n_blocks = get_num_tasks() - 1;
	SG_DEBUG("Number of blocks = {} ", n_blocks)

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
