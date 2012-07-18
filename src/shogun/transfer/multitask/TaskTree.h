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

#include <shogun/transfer/multitask/Task.h>
#include <shogun/transfer/multitask/TaskRelation.h>

namespace shogun
{

class CTaskTree : public CTaskRelation
{
public:

	/** default constructor */
	CTaskTree();

	/** constructor
	 * @param root_task root task of the tree
	 */
	CTaskTree(CTask* root_task);

	/** destructor */
	virtual ~CTaskTree();

	/** returns information about blocks in 
	 * SLEP "ind" format
	 */
	SGVector<index_t> get_SLEP_ind();

	/** returns information about blocks relations
	 * in SLEP "ind_t" format
	 */
	SGVector<float64_t> get_SLEP_ind_t();

	/** get root task */
	CTask* get_root_task() const { SG_REF(m_root_task); return m_root_task; }
	/** set root task */
	void set_root_task(CTask* root_task) { SG_REF(root_task); SG_UNREF(m_root_task); m_root_task = root_task; }

	/** get name */
	const char* get_name() const { return "TaskTree"; };

	/** get relation type */
	ETaskRelationType get_relation_type() const { return TASK_TREE; }

protected:

	/** root task */
	CTask* m_root_task;

};

}
#endif

