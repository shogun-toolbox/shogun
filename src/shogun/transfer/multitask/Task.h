/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#ifndef TASK_H_
#define TASK_H_

#include <shogun/lib/SGVector.h>
#include <shogun/lib/List.h>
#include <shogun/lib/IndexBlock.h>

namespace shogun
{

/** @brief used to represent tasks in multitask learning
 */
class CTask : public CIndexBlock
{
public:

	/** default constructor */
	CTask();

	/** constructor
	 * @param min_index smallest index of the task
	 * @param max_index largest index of the task
	 * @param weight weight (optional)
	 * @param name name of task (optional)
	 */
	CTask(index_t min_index, index_t max_index, 
	      float64_t weight=1.0, const char* name="task");

	/** destructor */
	~CTask();

	/** get name */
	virtual const char* get_name() const { return "Task"; };

};

}
#endif
