/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2012 Chiyuan Zhang
 * Written (w) 2014 Parijat Mazumdar
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 */

#ifndef TREEMACHINENODE_H__
#define TREEMACHINENODE_H__

#include <shogun/base/SGObject.h>
#include <shogun/base/Parameter.h>
#include <shogun/lib/DynamicObjectArray.h>

namespace shogun
{

/** @brief The node of the tree structure forming a TreeMachine
 * The node contains pointer to its parent and vector of
 * pointers to its children. A node of this class can have
 * only one parent but any number of children.The node also
 * contains data which can be of any type and has to be 
 * specified using template specifier. 
 */
template <typename T>
class CTreeMachineNode
	: public CSGObject
{
public:
	/** constructor */
	CTreeMachineNode() : CSGObject()
	{
		init();
	}

	/** destructor */
	virtual ~CTreeMachineNode()
	{
		SG_UNREF(m_parent)
		SG_UNREF(m_children);
	}

	/** get name
	 * @return class of the node 
	 */
	virtual const char* get_name() const { return "TreeMachineNode"; }

	/** set machine index
	 * @param idx the machine index
	 */
	void machine(int32_t idx)
	{
		m_machine=idx;
	}

	/** get machine
	 * @return machine index 
	 */
	int32_t machine()
	{
		return m_machine;
	}

	/** set parent node
	 * @param par parent node
	 */
	void parent(CTreeMachineNode* par)
	{
		SG_UNREF(m_parent);
		SG_REF(par);
		m_parent=par;
	}

	/** get parent node
	 * @return pointer to parent node 
	 */
	CTreeMachineNode* parent()
	{
		SG_REF(m_parent);
		return m_parent;
	}

	/** set children
	 * @param children dynamic array of pointers to children
	 */
	virtual void set_children(CDynamicObjectArray* children)
	{
		m_children->reset_array();
		for (int32_t i=0; i<children->get_num_elements(); i++)
			add_child((CTreeMachineNode*) children->get_element(i));
	}

	/** add child
	 * @param child pointer to child node
	 */
	virtual void add_child(CTreeMachineNode* child)
	{
		m_children->push_back(child);
		child->parent(this);
	}

	/** get children
	 * @return dynamic array of pointers to children
	 */
	virtual CDynamicObjectArray* get_children()
	{
		return m_children;
	}

	/** print function */
	typedef void (*data_print_func_t) (const T&);

	/** debug print the tree structure
	 * @param data_print_func the function to print the data payload
	 */
	void debug_print(data_print_func_t data_print_func)
	{
		debug_print_impl(data_print_func, this, 0);
	}

protected:
	/** implementation of printing the tree for debugging purpose
	 * @param data_print_func function for printing data
	 * @param node node data to print
	 * @param depth depth of the node in the tree
	 */
	static void debug_print_impl(data_print_func_t data_print_func, 
				CTreeMachineNode<T>* node, int32_t depth)
	{
		for (int32_t i=0;i<depth;++i)
			SG_SPRINT("  ");

		data_print_func(node->data);

		CDynamicObjectArray* childrenVector=node->get_children();
		for (int32_t j=0;j<childrenVector->get_num_elements();j++)
			debug_print_impl(data_print_func,(CTreeMachineNode<T>*)
					 childrenVector->get_element(j),depth+1);
	}

private:
	/* initialize parameters in constructor */
	void init()
	{
		m_parent=NULL;
		m_machine=-1;
		m_children=new CDynamicObjectArray();
		SG_ADD((CSGObject**)&m_parent,"m_parent", "Parent node", MS_NOT_AVAILABLE);
		SG_ADD(&m_machine,"m_machine", "Index of associated machine", MS_NOT_AVAILABLE);

	}

public:
	/** extra data carried by the tree node */
	T data;

protected:
	/* parent node */
	CTreeMachineNode* m_parent;

	/* machine index */
	int32_t m_machine;

	/* Dynamic array of pointers to children */ 
	CDynamicObjectArray* m_children;

};

} /* namespace shogun */

#endif /* end of include guard: TREEMACHINENODE_H__ */

