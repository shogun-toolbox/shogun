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

#include <lib/SGVector.h>
#include <lib/List.h>

namespace shogun
{

/** @brief class Task used to represent tasks in multitask learning.
 * Essentially it represent a set of feature vector indices.
 *
 * @see CTaskGroup
 * @see CTaskTree
 */
class CTask : public CSGObject
{
public:

	/** default constructor */
	CTask();

	/** constructor from range
	 * @param min_index smallest index of the task
	 * @param max_index largest index of the task
	 * @param weight weight (optional)
	 * @param name name of task (optional)
	 */
	CTask(index_t min_index, index_t max_index,
	      float64_t weight=1.0, const char* name="task");

	/** constructor from indices
	 * @param indices indices of the task
	 * @param weight weight (optional)
	 * @param name name of task (optional)
	 */
	CTask(SGVector<index_t> indices, float64_t weight=1.0, const char* name="task");

	/** destructor */
	virtual ~CTask();

	/** is contiguous
	 *
	 * @return whether task is contiguous
	 */
	bool is_contiguous();

	/** get indices
	 *
	 * @return indices
	 */
	SGVector<index_t> get_indices() const { return m_indices; }

	/** set indices
	 *
	 * @param indices task vector indices to set
	 */
	void set_indices(SGVector<index_t> indices) { m_indices = indices; }

	/** get weight of the task
	 *
	 * @return weight of the task
	 */
	float64_t get_weight() const { return m_weight; }

	/** set weight of the task
	 *
	 * @param weight weight of the task
	 */
	void set_weight(float64_t weight) { m_weight = weight; }

	/** get task name
	 *
	 * @return name of the task
	 */
	const char* get_task_name() const { return m_name; }

	/** set task name
	 *
	 * @param name name of the task
	 */
	void set_task_name(const char* name) { m_name = name; }

	/** add subtask
	 * should represent a subset of indices of the task
	 *
	 * @param sub_task subtask to add
	 */
	void add_subtask(CTask* sub_task);

	/** get all subtasks of the task
	 *
	 * @return subtasks of the task
	 */
	CList* get_subtasks();

	/** get number of subtasks
	 *
	 * @return number of subtasks
	 */
	int32_t get_num_subtasks();

	/** get name
	 *
	 * @return name of object
	 */
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
