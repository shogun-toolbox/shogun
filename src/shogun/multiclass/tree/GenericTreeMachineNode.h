/*
 * Copyright (c) The Shogun Machine Learning Toolbox
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

#ifndef GENERICTREEMACHINENODE_H__
#define GENERICTREEMACHINENODE_H__

#include <shogun/base/SGObject.h>
#include <shogun/base/Parameter.h>
#include <vector>

using namespace std;

namespace shogun
{

/** @brief The node of the tree structure forming a GenericTreeMachine */
template <typename T>
class CGenericTreeMachineNode
 : public CSGObject
{
public:
	/** constructor */
	CGenericTreeMachineNode():CSGObject()
	{
		m_parent=NULL;
		m_machine=-1;
		SG_ADD((CSGObject**)&m_parent,"m_parent", "Parent node", MS_NOT_AVAILABLE);
		SG_ADD(&m_machine,"m_machine", "Index of associated machine", MS_NOT_AVAILABLE);
	}

	/** destructor */
	virtual ~CGenericTreeMachineNode()
	{
	}

	/** get name */
	virtual const char* get_name() const { return "GenericTreeMachineNode"; }

	/** set machine index
	 * @param idx the machine index
	 */
	void set_machine_index(int32_t idx)
	{
		m_machine=idx;
	}

	/** get machine
	 * @return machine index 
	 */
	int32_t get_machine_index()
	{
		return m_machine;
	}

	/** set parent node
	 * @param par parent node
	 */
	void set_parent(CGenericTreeMachineNode* par)
	{
		m_parent=par;
	}
	/** get parent node
	 * @return pointer to parent node 
	 */
	CGenericTreeMachineNode* get_parent()
	{
		return m_parent;
	}

	/** set children
	 * @param children vector of pointers to children
	 */
	void set_children(CGenericTreeMachineNode** children, int32_t num_children)
	{
		m_children.clear();

		for (int32_t i=0;i<num_children;i++)
			m_children.push_back(children[i]);
	}

	/** add child
	 * @param child pointer to child node
	 */
	void add_child(CGenericTreeMachineNode* child)
	{
		m_children.push_back(child);
	}

	/** get children
	 * @return vector of pointers to children
	 */
	vector<CGenericTreeMachineNode*> get_children()
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
	static void debug_print_impl(data_print_func_t data_print_func, 
				CGenericTreeMachineNode<T>* node, int32_t depth)
	{
		for (int32_t i=0; i < depth; ++i)
			SG_SPRINT("  ");

		data_print_func(node->data);

		vector<CGenericTreeMachineNode*> childrenVector(node->get_children());
		for (int32_t j=0;j<childrenVector.size();j++)
			debug_print_impl(data_print_func, childrenVector[j], depth+1);
	}

	/* parent node */
	CGenericTreeMachineNode* m_parent;

	/* vector of pointers to children */ 
	vector<CGenericTreeMachineNode*> m_children;

	/* machine index */
	int32_t m_machine;
};

} /* shogun */

#endif /* GENERICTREEMACHINENODE_H__ */
