/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Sergey Lisitsyn
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#ifndef INDICES_TREE_H_
#define INDICES_TREE_H_

#include <shogun/base/SGObject.h>

#include <shogun/lib/SGVector.h>
#include <shogun/lib/v_array.h>

namespace shogun
{

/** @brief indices tree node */
class CIndicesTreeNode
{
public:
	CIndicesTreeNode()
	{
		node_indices = SGVector<index_t>();
		child_nodes = v_array<CIndicesTreeNode*>();
	}

	CIndicesTreeNode(SGVector<index_t> indices)
	{
		node_indices = indices;
		child_nodes = v_array<CIndicesTreeNode*>();
	}

	~CIndicesTreeNode()
	{
		int32_t len_child_nodes = child_nodes.index();
		for (int32_t i; i<len_child_nodes; i++)
			delete child_nodes[i];

	}

	void add_child(CIndicesTreeNode* child)
	{
		child_nodes.push(child);
	}

	void clear_childs()
	{
		child_nodes.erase();
	}

	SGVector<index_t> node_indices;

	v_array<CIndicesTreeNode*> child_nodes;

};


class CIndicesTree : public CSGObject
{
public:
	CIndicesTree() : CSGObject()
	{
		root_node = new CIndicesTreeNode();
		current_node = root_node;
		last_node = current_node;
	}

	virtual ~CIndicesTree()
	{
		delete root_node;
	}

	float64_t* get_ind() const;

	void add_child(SGVector<index_t> indices)
	{
		current_node->add_child(new CIndicesTreeNode(indices));
	}

	void go_child(int32_t child_index)
	{
		last_node = current_node;
		current_node = current_node->child_nodes[child_index];
	}

	void go_back()
	{
		current_node = last_node;
	}

	virtual const char* get_name() const 
	{
		return "IndicesTree";
	}

private:

	CIndicesTreeNode* root_node;

	CIndicesTreeNode* current_node;

	CIndicesTreeNode* last_node;

};
}
#endif   /* ----- #ifndef INDICES_TREE_H_  ----- */
