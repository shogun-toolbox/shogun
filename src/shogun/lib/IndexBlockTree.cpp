/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#include <shogun/lib/IndexBlockTree.h>
#include <vector>

using namespace std;
using namespace shogun;

struct block_tree_node_t
{
	block_tree_node_t(int32_t min, int32_t max, float64_t w)
	{
		t_min_index = min;
		t_max_index = max;
		weight = w;
	}
	int32_t t_min_index, t_max_index;
	float64_t weight;
};

int32_t count_leaf_blocks_recursive(CIndexBlock* subtree_root_block)
{
	CList* sub_blocks = subtree_root_block->get_sub_blocks();
	int32_t n_sub_blocks = sub_blocks->get_num_elements();
	if (n_sub_blocks==0)
	{
		SG_UNREF(sub_blocks);
		return 1;
	}
	else
	{
		int32_t sum = 0;
		CIndexBlock* iterator = (CIndexBlock*)sub_blocks->get_first_element();
		do
		{
			sum += count_leaf_blocks_recursive(iterator);
		}
		while ((iterator = (CIndexBlock*)sub_blocks->get_next_element()) != NULL);

		SG_UNREF(sub_blocks);
		return sum;
	}
}

void collect_tree_nodes_recursive(CIndexBlock* subtree_root_block, vector<block_tree_node_t>* tree_nodes, int low)
{
	int32_t lower = low;
	CList* sub_blocks = subtree_root_block->get_sub_blocks();
	if (sub_blocks->get_num_elements()>0)
	{
		CIndexBlock* iterator = (CIndexBlock*)sub_blocks->get_first_element();
		do
		{
			if (iterator->get_num_sub_blocks()>0)
			{
				int32_t n_leaves = count_leaf_blocks_recursive(iterator);
				SG_SDEBUG("Block [%d %d] has %d leaf childs \n",iterator->get_min_index(), iterator->get_max_index(), n_leaves);
				tree_nodes->push_back(block_tree_node_t(lower,lower+n_leaves-1,iterator->get_weight()));
				collect_tree_nodes_recursive(iterator, tree_nodes, lower);
				lower = lower + n_leaves;
			}
			else
				lower++;
			SG_UNREF(iterator);
		}
		while ((iterator = (CIndexBlock*)sub_blocks->get_next_element()) != NULL);
	}
	SG_UNREF(sub_blocks);
}

void collect_leaf_blocks_recursive(CIndexBlock* subtree_root_block, CList* list)
{
	CList* sub_blocks = subtree_root_block->get_sub_blocks();
	if (sub_blocks->get_num_elements() == 0)
	{
		list->append_element(subtree_root_block);
	}
	else
	{
		CIndexBlock* iterator = (CIndexBlock*)sub_blocks->get_first_element();
		do
		{
			collect_leaf_blocks_recursive(iterator, list);
			SG_UNREF(iterator);
		} 
		while ((iterator = (CIndexBlock*)sub_blocks->get_next_element()) != NULL);
	}
	SG_UNREF(sub_blocks);
}

CIndexBlockTree::CIndexBlockTree() : CIndexBlockRelation(), m_root_block(NULL)
{

}

CIndexBlockTree::CIndexBlockTree(CIndexBlock* root_block) : CIndexBlockRelation(),
	m_root_block(NULL)
{
	set_root_block(root_block);
}

CIndexBlockTree::~CIndexBlockTree()
{
	SG_UNREF(m_root_block);
}

CIndexBlock* CIndexBlockTree::get_root_block() const
{
	SG_REF(m_root_block);
	return m_root_block;
}

void CIndexBlockTree::set_root_block(CIndexBlock* root_block)
{
	SG_REF(root_block);
	SG_UNREF(m_root_block);
	m_root_block = root_block;
}

SGVector<index_t> CIndexBlockTree::get_SLEP_ind()
{
	CList* blocks = new CList(true);
	collect_leaf_blocks_recursive(m_root_block, blocks);
	SG_DEBUG("Collected %d leaf blocks\n", blocks->get_num_elements());
	check_blocks_list(blocks);


	SGVector<index_t> ind(blocks->get_num_elements()+1);

	int t_i = 0;
	ind[0] = 0;
	CIndexBlock* iterator = (CIndexBlock*)blocks->get_first_element();
	do
	{
		ind[t_i+1] = iterator->get_max_index();
		SG_DEBUG("Blocks = [%d,%d]\n", iterator->get_min_index(), iterator->get_max_index());
		SG_UNREF(iterator);
		t_i++;
	} 
	while ((iterator = (CIndexBlock*)blocks->get_next_element()) != NULL);

	SG_UNREF(blocks);

	return ind;
}

SGVector<float64_t> CIndexBlockTree::get_SLEP_ind_t()
{
	CList* blocks = new CList(true);
	int n_blocks = get_SLEP_ind().vlen;
	SG_DEBUG("Number of blocks = %d \n", n_blocks);

	vector<block_tree_node_t> tree_nodes = vector<block_tree_node_t>();
	
	collect_tree_nodes_recursive(m_root_block, &tree_nodes,1);

	SGVector<float64_t> ind_t(3+3*tree_nodes.size());
	// supernode
	ind_t[0] = -1;
	ind_t[1] = -1;
	ind_t[2] = 1.0;

	for (int32_t i=0; i<tree_nodes.size(); i++)
	{
		ind_t[3+i*3] = tree_nodes[i].t_min_index;
		ind_t[3+i*3+1] = tree_nodes[i].t_max_index;
		ind_t[3+i*3+2] = tree_nodes[i].weight;
	}

	SG_UNREF(blocks);

	return ind_t;
}
