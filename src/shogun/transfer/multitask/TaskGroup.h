/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#ifndef TASKGROUP_H_
#define TASKGROUP_H_

#include <shogun/transfer/multitask/Task.h>
#include <shogun/transfer/multitask/TaskRelation.h>

namespace shogun
{

/** @brief
 *
 */
class CTaskGroup : public CTaskRelation
{
public:

	/** default constructor */
	CTaskGroup();

	/** destructor */
	virtual ~CTaskGroup();

	/** add task to group
	 * @param task task to add
	 */
	void add_task(CTask* task);

	/** returns information about tasks in 
	 * SLEP "ind" format
	 */
	SGVector<index_t> get_SLEP_ind();

	virtual bool is_valid() const;

	/** get name */
	const char* get_name() const { return "TaskGroup"; };

protected:

	/** tasks in group */
	CList* m_tasks;

};

}
#endif

