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
	 * @param root_task root task of task tree
	 */
	CTaskTree(CTask* root_task);

	/** destructor */
	virtual ~CTaskTree();

	/** get root task */
	CTask* get_root_task() const;

	/** set root task */
	void set_root_task(CTask* root_task);

	/** returns information about tasks in 
	 * SLEP "ind" format
	 */
	SGVector<index_t> get_SLEP_ind();

	/** returns information about tasks relationships
	 * in SLEP "ind_t" format
	 */
	SGVector<float64_t> get_SLEP_ind_t();

	virtual bool is_valid() const;

	virtual ETaskRelationType get_relation_type() const { return TREE; }

	/** get name */
	const char* get_name() const { return "TaskTree"; };

private:

	/** root task */
	CTask* m_root_task;
};

}
#endif

