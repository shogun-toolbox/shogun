/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Yuyu Zhang, Bjoern Esser
 */

#ifndef TASKGROUP_H_
#define TASKGROUP_H_

#include <shogun/lib/config.h>

#include <shogun/base/SGObject.h>
#include <shogun/transfer/multitask/Task.h>
#include <shogun/transfer/multitask/TaskRelation.h>

namespace shogun
{

/** @brief class TaskGroup used to represent a group of tasks.
 * Tasks in group do not overlap.
 *
 * @see CTask
 */
class TaskGroup : public TaskRelation
{
public:

	/** default constructor */
	TaskGroup();

	/** destructor */
	~TaskGroup() override;

	/** get tasks indices
	 *
	 * @return array of vectors containing indices of each task
	 */
	SGVector<index_t>* get_tasks_indices() const override;

	/** append task to the group
	 *
	 * @param task task to append
	 */
	void append_task(std::shared_ptr<Task> task);

	/** get number of tasks in the group
	 *
	 * @return number of tasks in the group
	 */
	int32_t get_num_tasks() const override;

	/** get name
	 *
	 * @return name of the object
	 */
	const char* get_name() const override { return "TaskGroup"; };

	/** get relation type
	 *
	 * @return TASK_GROUP
	 */
	ETaskRelationType get_relation_type() const override { return TASK_GROUP; }

private:

	/** init */
	void init();

protected:

	/** tasks of the task group */
	std::vector<std::shared_ptr<Task>> m_tasks;

};
}
#endif

