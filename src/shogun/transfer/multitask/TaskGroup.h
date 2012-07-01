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

/** @brief contains a few of tasks of multitask context
 *
 * For proper numbering of tasks in multitask-kind machine
 * please add tasks to the group in well-ordered fashion.
 *
 * Example of usage: with 10 vectors available consider
 * first 5 vectors belong to the first task and latter 5
 * vectors belong to the second task - task group should
 * be constructed via two CTask instances - CTask(0,5) and 
 * CTask(5,10).
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

	/** remove task from group
	 * @param task task to remove
	 */
	void remove_task(CTask* task);

	/** returns information about tasks in 
	 * SLEP "ind" format
	 */
	SGVector<index_t> get_SLEP_ind();

	virtual bool is_valid() const;

	virtual ETaskRelationType get_relation_type() const { return GROUP; }

	/** get name */
	const char* get_name() const { return "TaskGroup"; };

protected:

	/** tasks in group */
	CList* m_tasks;

};

}
#endif

