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

namespace shogun
{

/** @brief used to represent tasks in multitask learning
 */
class CTask : public CSGObject
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
	
	/** constructor
	 * @param indices indices of the task
	 * @param weight weight (optional)
	 * @param name name of task (optional)
	 */
	CTask(SGVector<index_t> indices, float64_t weight=1.0, const char* name="task");

	/** destructor */
	virtual ~CTask();

	/** is contiguous */
	bool is_contiguous();

	/** get indices */
	SGVector<index_t> get_indices() const { return m_indices; }
	/** set indices */
	void set_indices(SGVector<index_t> indices) { m_indices = indices; }

	/** get weight */
	float64_t get_weight() const { return m_weight; }
	/** set weight */
	void set_weight(float64_t weight) { m_weight = weight; }
	/** get task name */
	const char* get_task_name() const { return m_name; }
	/** set task name */
	void set_task_name(const char* name) { m_name = name; }

	/** add sub task */
	void add_subtask(CTask* sub_task);
	/** get subtasks */
	CList* get_subtasks();
	/** get num subtasks */
	int32_t get_num_subtasks();

	/** get name */
	virtual const char* get_name() const { return "Task"; };

private:

	/** init */
	void init();

protected:

	/** subtasks */
	CList* m_subtasks;

	/** name of the block */
	const char* m_name;

	/** indices */
	SGVector<index_t> m_indices;

	/** weight */
	float64_t m_weight;

};

}
#endif
