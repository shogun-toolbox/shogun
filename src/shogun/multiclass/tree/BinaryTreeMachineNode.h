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

#ifndef BINARYTREEMACHINENODE_H__
#define BINARYTREEMACHINENODE_H__

#include <shogun/base/SGObject.h>
#include <shogun/base/Parameter.h>
#include <shogun/multiclass/tree/TreeMachineNode.h>

namespace shogun
{

/** @brief The node of the tree structure forming a TreeMachine
 * The node contains pointer to its parent and pointers to its
 * 2 children: left child and right child. The node also
 * contains data which can be of any type and has to be 
 * specified using template specifier. 
 */
template <typename T>
class CBinaryTreeMachineNode
	: public CTreeMachineNode<T>
{
public:
	/** constructor */
	CBinaryTreeMachineNode() : CTreeMachineNode<T>()
	{
	}


	/** destructor */
	virtual ~CBinaryTreeMachineNode()
	{
	}

	/** get name
	 * @return class of the node 
	 */
	virtual const char* get_name() const { return "BinaryTreeMachineNode"; }

	/** set left subtree
	 *
	 * @param l left subtree
	 */
	void left(CBinaryTreeMachineNode* l)
	{
		if (this->m_children->get_num_elements()==0)
		{
			this->m_children->push_back(l);
			l->parent(this);
		}
		else
		{
			this->m_children->set_element(l,0);
			l->parent(this);
		}
	}

	/** get left subtree
	 *
	 * @return left subtree of node
	 */
	CBinaryTreeMachineNode* left()
	{
		if (this->m_children->get_num_elements())
			return (CBinaryTreeMachineNode*) this->m_children->get_element(0);

		return NULL;
	}

	/** set right subtree
	 *
	 * @param r right subtree
	 */
	void right(CBinaryTreeMachineNode* r)
	{
		if (this->m_children->get_num_elements()==0)
		{
			this->m_children->push_back(NULL);
			this->m_children->push_back(r);
			r->parent(this);
		}
		else if (this->m_children->get_num_elements()==1)
		{
			this->m_children->push_back(r);
			r->parent(this);
		}
		else
		{
			this->m_children->set_element(r,1);
			r->parent(this);
		}
	}

	/** get right subtree
	 *
	 * @return right subtree of node 
	 */
	CBinaryTreeMachineNode* right()
	{
		if (this->m_children->get_num_elements()==2)
			return (CBinaryTreeMachineNode*) this->m_children->get_element(1);

		return NULL;
	}

};

} /* namespace shogun */

#endif /* BINARYTREEMACHINENODE_H__ */

