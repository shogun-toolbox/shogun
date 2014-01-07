/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#ifndef TREEMACHINENODE_H__
#define TREEMACHINENODE_H__

#include <base/SGObject.h>
#include <base/Parameter.h>

namespace shogun
{

/** The node of the tree structure forming a TreeMachine */
template <typename T>
class CTreeMachineNode
	: public CSGObject
{
public:
    /** constructor */
	CTreeMachineNode()
		:m_left(NULL), m_right(NULL), m_parent(NULL), m_machine(-1)
	{
		SG_ADD((CSGObject**)&m_left,"m_left", "Left subtree", MS_NOT_AVAILABLE);
		SG_ADD((CSGObject**)&m_right,"m_right", "Right subtree", MS_NOT_AVAILABLE);
		SG_ADD((CSGObject**)&m_parent,"m_parent", "Parent node", MS_NOT_AVAILABLE);
		SG_ADD(&m_machine,"m_machine", "Index of associated machine", MS_NOT_AVAILABLE);
	}


    /** destructor */
	virtual ~CTreeMachineNode()
	{
		SG_UNREF(m_left);
		SG_UNREF(m_right);
	}

    /** get name */
    virtual const char* get_name() const { return "TreeMachineNode"; }

	/** set machine index
	 * @param idx the machine index
	 */
	void machine(int32_t idx)
	{
		m_machine = idx;
	}
	/** get machine */
	int32_t machine()
	{
		return m_machine;
	}

	/** set parent node
	 * @param par parent node
	 */
	void parent(CTreeMachineNode *par)
	{
		m_parent = par;
	}
	/** get parent node */
	CTreeMachineNode *parent()
	{
		return m_parent;
	}

	/** set left subtree
	 * @param l left subtree
	 */
	void left(CTreeMachineNode *l)
	{
		SG_REF(l);
		SG_UNREF(m_left);
		m_left = l;
		m_left->parent(this);
	}
	/** get left subtree */
	CTreeMachineNode *left()
	{
		return m_left;
	}

	/** set right subtree
	 * @param r right subtree
	 */
	void right(CTreeMachineNode *r)
	{
		SG_REF(r);
		SG_UNREF(m_right);
		m_right = r;
		m_right->parent(this);
	}
	/** get right subtree */
	CTreeMachineNode *right()
	{
		return m_right;
	}

	/** extra data carried by the tree node */
	T data;

	/** print function */
	typedef void (*data_print_func_t) (const T&);
	/** debug print the tree structure
	 * @param data_print_func the function to print the data payload
	 */
	void debug_print(data_print_func_t data_print_func)
	{
		debug_print_impl(data_print_func, this, 0);
	}

private:
	CTreeMachineNode *m_left;    ///< left subtree
	CTreeMachineNode *m_right;   ///< right subtree
	CTreeMachineNode *m_parent;  ///< parent node
	int32_t           m_machine; ///< machine index associated with this node

    /** implementation of printing the tree for debugging purpose */
	static void debug_print_impl(data_print_func_t data_print_func, CTreeMachineNode<T> *node, int32_t depth)
	{
		for (int32_t i=0; i < depth; ++i)
			SG_SPRINT("  ")
		data_print_func(node->data);
		if (node->left())
			debug_print_impl(data_print_func, node->left(), depth+1);
		if (node->right())
			debug_print_impl(data_print_func, node->right(), depth+1);
	}
};

} /* shogun */

#endif /* end of include guard: TREEMACHINENODE_H__ */

