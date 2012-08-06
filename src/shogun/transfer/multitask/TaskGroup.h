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

class CTaskGroup : public CTaskRelation
{
public:

	/** default constructor */
	CTaskGroup();

	/** destructor */
	virtual ~CTaskGroup();

	/** returns information about tasks
	 */
	virtual SGVector<index_t>* get_tasks_indices() const;

	/** append task to the group */
	void append_task(CTask* task);
	
	/** get num tasks */
	virtual int32_t get_num_tasks() const;

	/** get name */
	const char* get_name() const { return "TaskGroup"; };

	/** relation type */
	ETaskRelationType get_relation_type() const { return TASK_GROUP; }

private:

	void init();

protected:

	/** tasks */
	CDynamicObjectArray* m_tasks;

};

}
#endif

