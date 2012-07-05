/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#ifndef TASKTREE_H_
#define TASKTREE_H_

#include <shogun/lib/IndexBlockTree.h>

namespace shogun
{

class CTaskTree : public CIndexBlockTree
{
public:

	/** default constructor */
	CTaskTree();

	/** constructor
	 * @param root_task root task of the tree
	 */
	CTaskTree(CIndexBlock* root_task);

	/** destructor */
	virtual ~CTaskTree();

	/** returns information about blocks in 
	 * SLEP "ind" format
	 */
	virtual SGVector<index_t> get_SLEP_ind();

	/** returns information about blocks relations
	 * in SLEP "ind_t" format
	 */
	virtual SGVector<float64_t> get_SLEP_ind_t();

	virtual EIndexBlockRelationType get_relation_type() const { return TREE; }

	/** get name */
	const char* get_name() const { return "TaskTree"; };

};

}
#endif

