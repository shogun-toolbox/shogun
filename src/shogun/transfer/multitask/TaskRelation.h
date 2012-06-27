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
#include <shogun/lib/List.h>

namespace shogun
{

enum ETaskRelationType
{
	GROUP,
	TREE
};

/** @brief
 *
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
	const char* get_name() const { return "TaskRelation"; };

	/** check validity of relation */
	virtual bool is_valid() const = 0;

	/** get relation type */
	virtual ETaskRelationType get_relation_type() const = 0;

protected:

	bool check_task_list(CList* tasks);

};

}
#endif

