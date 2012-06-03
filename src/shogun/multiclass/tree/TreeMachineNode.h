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

#include <shogun/base/SGObject.h>
#include <shogun/base/Parameter.h>

namespace shogun
{

class CTreeMachineNode: public CSGObject
{
public:
    /** constructor */
	CTreeMachineNode();

    /** destructor */
	virtual ~CTreeMachineNode();

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

private:
	CTreeMachineNode *m_left;    ///< left subtree
	CTreeMachineNode *m_right;   ///< right subtree
	CTreeMachineNode *m_parent;  ///< parent node
	int32_t           m_machine; ///< machine index associated with this node
};

} /* shogun */ 

#endif /* end of include guard: TREEMACHINENODE_H__ */

