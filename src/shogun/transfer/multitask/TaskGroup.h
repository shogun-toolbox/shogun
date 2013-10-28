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

#include <shogun/base/SGObject.h>
#include <shogun/transfer/multitask/Task.h>
#include <shogun/transfer/multitask/TaskRelation.h>
#include <shogun/lib/DynamicObjectArray.h>

namespace shogun
{

/** @brief class TaskGroup used to represent a group of tasks.
 * Tasks in group do not overlap.
 *
 * @see CTask
 */
class CTaskGroup : public CTaskRelation
{
public:

	/** default constructor */
	CTaskGroup();

	/** destructor */
	virtual ~CTaskGroup();

	/** get tasks indices
	 *
	 * @return array of vectors containing indices of each task
	 */
	virtual SGVector<index_t>* get_tasks_indices() const;

	/** append task to the group
	 *
	 * @param task task to append
	 */
	void append_task(CTask* task);

	/** get number of tasks in the group
	 *
	 * @return number of tasks in the group
	 */
	virtual int32_t get_num_tasks() const;

	/** get name
	 *
	 * @return name of the object
	 */
	const char* get_name() const { return "TaskGroup"; };

	/** get relation type
	 *
	 * @return TASK_GROUP
	 */
	ETaskRelationType get_relation_type() const { return TASK_GROUP; }

private:

	/** init */
	void init();

protected:

	/** tasks of the task group */
	CDynamicObjectArray* m_tasks;

};
}
#endif

