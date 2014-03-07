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
#include <vector>

using namespace std;

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
	CTreeMachineNode():CSGObject()
	{
		m_parent=NULL;
		m_machine=-1;
		m_children=vector<CTreeMachineNode*>();
		SG_ADD((CSGObject**)&m_parent,"m_parent", "Parent node", MS_NOT_AVAILABLE);
		SG_ADD(&m_machine,"m_machine", "Index of associated machine", MS_NOT_AVAILABLE);
	}


	/** destructor */
	virtual ~CTreeMachineNode()
	{
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
		m_machine = idx;
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
	void parent(CTreeMachineNode *par)
	{
		m_parent = par;
	}

	/** get parent node
	 * @return pointer to parent node 
	 */
	CTreeMachineNode *parent()
	{
		return m_parent;
	}

	/** set left subtree
	 * method is valid only if node has max 2 children
	 *
	 * @param l left subtree
	 */
	void left(CTreeMachineNode *l)
	{
		REQUIRE(m_children.size()<3,
				"Number of children already greater than 2")
		SG_REF(l);

		if (m_children.size()==0)
		{
			add_child(l);
		}

		else
		{
			m_children[0]=l;
			l->parent(this);
		}
	}

	/** get left subtree
	 * method is valid only if node has max 2 children
	 *
	 * @return left subtree of node
	 */
	CTreeMachineNode *left()
	{
		REQUIRE(m_children.size()<3,"Number of children can be max 2")

		if (m_children.size())
			return m_children[0];
		return NULL;
	}

	/** set right subtree
	 * method is valid only if node has max 2 children
	 *
	 * @param r right subtree
	 */
	void right(CTreeMachineNode *r)
	{
		REQUIRE(m_children.size()<3,
				"Number of children already greater than 2")
		SG_REF(r);

		if (m_children.size()==0)
		{
			m_children.push_back(NULL);
			add_child(r);
		}

		else if (m_children.size()==1)
		{
			add_child(r);
		}

		else
		{
			m_children[1]=r;
			r->parent(this);
		}
	}

	/** get right subtree
	 * method is valid only if node has max 2 children
	 *
	 * @return right subtree of node 
	 */
	CTreeMachineNode *right()
	{
		REQUIRE(m_children.size()<3,"Number of children can be max 2")

		if (m_children.size()==2)
			return m_children[1];
		return NULL;
	}

	/** set children
	 * @param children vector of pointers to children
	 * @param num_children number of children
	 */
	void set_children(CTreeMachineNode** children, int32_t num_children)
	{
		m_children.clear();

		for (int32_t i=0;i<num_children;i++)
			add_child(children[i]);
	}

	/** add child
	 * @param child pointer to child node
	 */
	void add_child(CTreeMachineNode* child)
	{
		m_children.push_back(child);
		child->parent(this);
	}

	/** get children
	 * @return vector of pointers to children
	 */
	vector<CTreeMachineNode*> get_children()
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

	/** extra data carried by the tree node */
	T data;

private:
	/** implementation of printing the tree for debugging purpose */
	/** implementation of printing the tree for debugging purpose
	 * @param data_print_func function for printing data
	 * @param node node data to print
	 * @param depth depth of the node in the tree
	 */
	static void debug_print_impl(data_print_func_t data_print_func, 
				CTreeMachineNode<T>* node, int32_t depth)
	{
		for (int32_t i=0; i < depth; ++i)
			SG_SPRINT("  ");

		data_print_func(node->data);

		vector<CTreeMachineNode*> childrenVector(node->get_children());
		for (int32_t j=0;j<childrenVector.size();j++)
			debug_print_impl(data_print_func, childrenVector[j], depth+1);
	}

	/* parent node */
	CTreeMachineNode* m_parent;

	/* vector of pointers to children */ 
	vector<CTreeMachineNode*> m_children;

	/* machine index */
	int32_t m_machine;
};

} /* shogun */

#endif /* end of include guard: TREEMACHINENODE_H__ */

