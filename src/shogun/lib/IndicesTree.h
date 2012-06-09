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

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/** @brief indices tree node */
class CIndicesTreeNode
{
public:
	CIndicesTreeNode()
	{
		node_indices = SGVector<index_t>();
		child_nodes = v_array<CIndicesTreeNode*>();
		weight = 0.0;
	}

	CIndicesTreeNode(SGVector<index_t> indices, float64_t w)
	{
		node_indices = indices;
		child_nodes = v_array<CIndicesTreeNode*>();
		weight = w;
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

	float64_t weight;

	v_array<CIndicesTreeNode*> child_nodes;

};
#endif

/** @brief indices tree
 *
 */
class CIndicesTree : public CSGObject
{
public:
	
	/** constructor */
	CIndicesTree() : CSGObject()
	{
		SGVector<int32_t> supernode(2);
		supernode[0] = -1;
		supernode[1] = -1;
		m_root_node = new CIndicesTreeNode(supernode,1.0);
		m_current_node = m_root_node;
		m_last_node = m_root_node;
		m_num_nodes = 1;
	}

	/** destructor */
	virtual ~CIndicesTree()
	{
		delete m_root_node;
	}

	/** get indices in SLEP format 
	 * @return indices
	 */
	SGVector<float64_t> get_ind() const;

	/** add child
	 *
	 * @param indices indices to add to the tree as child
	 */
	void add_child(SGVector<index_t> indices, float64_t weight)
	{
		m_num_nodes++;
		m_current_node->add_child(new CIndicesTreeNode(indices,weight));
	}

	/** move to specific child of current node
	 *
	 * @param child_index index of child
	 */
	void go_child(int32_t child_index)
	{
		m_last_node = m_current_node;
		m_current_node = m_current_node->child_nodes[child_index];
	}

	/** move back
	 *
	 */
	void go_back()
	{
		CIndicesTreeNode* current_node = m_current_node;
		m_current_node = m_last_node;
		m_last_node = current_node;
	}

	/** move to root
	 *
	 */
	void go_root()
	{
		m_last_node = m_current_node;
		m_current_node = m_root_node;
	}

	/** get number of nodes
	 *
	 */
	inline int32_t get_num_nodes() const
	{
		return m_num_nodes;
	}

	/** print tree */
	void print_tree() const;

	/** get name */
	virtual const char* get_name() const 
	{
		return "IndicesTree";
	}

protected:

	/** print tree recursive */
	void print_tree_recursive(CIndicesTreeNode* node, int32_t level) const;

private:

	/** root node */
	CIndicesTreeNode* m_root_node;

	/** current node */
	CIndicesTreeNode* m_current_node;

	/** last node */
	CIndicesTreeNode* m_last_node;

	/** number of nodes */
	int32_t m_num_nodes;

};
}
#endif   /* ----- #ifndef INDICES_TREE_H_  ----- */
