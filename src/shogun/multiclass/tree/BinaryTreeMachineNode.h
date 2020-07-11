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

#include <shogun/lib/config.h>

#include <shogun/base/SGObject.h>
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
class BinaryTreeMachineNode
	: public TreeMachineNode<T>
{
public:
	/** constructor */
	BinaryTreeMachineNode() : TreeMachineNode<T>()
	{
	}


	/** destructor */
	~BinaryTreeMachineNode() override
	{
	}

	/** get name
	 * @return class of the node
	 */
	const char* get_name() const override { return "BinaryTreeMachineNode"; }

	/** set left subtree
	 *
	 * @param l left subtree
	 */
	void left(std::shared_ptr<BinaryTreeMachineNode> l)
	{
		if (this->m_children.empty())
		{
			this->m_children.push_back(l);
		}
		else
		{
			this->m_children[0] = l;
		}
		l->parent(this->shared_from_this()->template as<TreeMachineNode<T>>());
	}

	/** get left subtree
	 *
	 * @return left subtree of node
	 */
	std::shared_ptr<BinaryTreeMachineNode<T>> left()
	{
		if (!this->m_children.empty())
			return this->m_children[0]->template as<BinaryTreeMachineNode<T>>();

		return NULL;
	}

	/** set right subtree
	 *
	 * @param r right subtree
	 */
	void right(std::shared_ptr<BinaryTreeMachineNode> r)
	{
		if (this->m_children.empty())
		{
			this->m_children.push_back(NULL);
			this->m_children.push_back(r);
		}
		else if (this->m_children.size()==1)
		{
			this->m_children.push_back(r);
		}
		else
		{
			this->m_children[1] = r;
		}
		r->parent(this->shared_from_this()->template as<TreeMachineNode<T>>());
	}

	/** get right subtree
	 *
	 * @return right subtree of node
	 */
	std::shared_ptr<BinaryTreeMachineNode<T>> right()
	{
		if (this->m_children.size()==2)
			return this->m_children[1]->template as<BinaryTreeMachineNode<T>>();

		return NULL;
	}

};

} /* namespace shogun */

#endif /* BINARYTREEMACHINENODE_H__ */

