/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#ifndef TASKRELATION_H_
#define TASKRELATION_H_

#include <shogun/base/SGObject.h>

namespace shogun
{

enum ETaskRelationType
{
	TASK_TREE,
	TASK_GROUP
};

/** @brief used to represent tasks in multitask learning
 */
class CTaskRelation : public CSGObject
{
public:

	/** default constructor */
	CTaskRelation()
	{
	}

	/** destructor */
	virtual ~CTaskRelation()
	{
	}
	
	/** get name */
	virtual const char* get_name() const { return "TaskRelation"; };

	/** get relation type */
	virtual ETaskRelationType get_relation_type() const = 0;

	/** get tasks indices */
	virtual SGVector<index_t>* get_tasks_indices() const = 0;

	/** get num tasks */
	virtual int32_t get_num_tasks() const = 0;
};

}
#endif
