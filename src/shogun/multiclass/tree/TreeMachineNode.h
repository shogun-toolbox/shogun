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

#include <shogun/lib/config.h>

#include <shogun/base/SGObject.h>

namespace shogun
{

/** @brief The node of the tree structure forming a TreeMachine
 * The node contains a pointer to its parent and a vector of
 * pointers to its children. A node of this class can have
 * only one parent but any number of children.The node also
 * contains data which can be of any type and has to be
 * specified using template specifier.
 */
template <typename T>
class TreeMachineNode
	: public SGObject
{
public:
	/** constructor */
	TreeMachineNode() : SGObject()
	{
		init();
	}

	/** destructor */
	virtual ~TreeMachineNode()
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
	void parent(std::shared_ptr<TreeMachineNode<T>> par)
	{
		m_parent=par;
	}

	/** get parent node
	 * @return pointer to parent node
	 */
	std::shared_ptr<TreeMachineNode<T>> parent()
	{
		return m_parent;
	}

	/** set children
	 * @param children dynamic array of pointers to children
	 */
	virtual void set_children(const std::vector<std::shared_ptr<TreeMachineNode<T>>>& children)
	{
		m_children.clear();
		m_children.reserve(children.size());
		for (auto& child : children)
		{
			add_child(child);
		}
	}

	/** add child
	 * @param child pointer to child node
	 */
	virtual void add_child(std::shared_ptr<TreeMachineNode<T>> child)
	{
		m_children.push_back(child);
		child->parent(shared_from_this()->template as<TreeMachineNode<T>>());
	}

	/** get children
	 * @return dynamic array of pointers to children
	 */
	virtual std::vector<std::shared_ptr<TreeMachineNode<T>>> get_children()
	{

		return m_children;
	}

	// FIXME: not the best idea to make this public
	// but at least then add a template for this...
	using SGObject::watch_param;

private:
	/** initialize parameters in constructor */
	void init()
	{
		m_machine=-1;
		m_children.clear();

		SG_ADD((std::shared_ptr<SGObject>*)&m_parent,"m_parent", "Parent node");
		SG_ADD(&m_machine,"m_machine", "Index of associated machine");
		register_params(data, this);
	}

public:
	/** extra data carried by the tree node */
	T data;

protected:
	/** parent node */
	std::shared_ptr<TreeMachineNode<T>> m_parent;

	/** machine index */
	int32_t m_machine;

	/** Dynamic array of pointers to children */
	std::vector<std::shared_ptr<TreeMachineNode<T>>> m_children;
};

} /* namespace shogun */

#endif /* end of include guard: TREEMACHINENODE_H__ */

