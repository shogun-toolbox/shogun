/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Yuyu Zhang, Bjoern Esser
 */

#ifndef TASKRELATION_H_
#define TASKRELATION_H_
#define IGNORE_IN_CLASSLIST

#include <shogun/lib/config.h>

#include <shogun/base/SGObject.h>

namespace shogun
{

#ifndef DOXYGEN_SHOULD_SKIP_THIS
IGNORE_IN_CLASSLIST enum ETaskRelationType
{
	TASK_TREE,
	TASK_GROUP
};
#endif

/** @brief used to represent tasks in multitask learning
 */
class TaskRelation : public SGObject
{
public:

	/** default constructor */
	TaskRelation()
	{
	}

	/** destructor */
	virtual ~TaskRelation()
	{
	}

	/** get name
	 *
	 * @return name of the object
	 */
	virtual const char* get_name() const { return "TaskRelation"; };

	/** get relation type (not implemented)
	 *
	 * @return type of relation
	 */
	virtual ETaskRelationType get_relation_type() const = 0;

	/** get tasks indices (not implemented)
	 *
	 * @return array of vectors containing indices of each task
	 */
	virtual SGVector<index_t>* get_tasks_indices() const = 0;

	/** get number of tasks in the group (not implemented)
	 *
	 * @return number of tasks in the group
	 */
	virtual int32_t get_num_tasks() const = 0;
};

}
#endif
