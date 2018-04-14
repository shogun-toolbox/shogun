/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Soeren Sonnenburg, Bjoern Esser
 */

#include <shogun/transfer/multitask/TaskGroup.h>

using namespace shogun;

CTaskGroup::CTaskGroup() : CTaskRelation()
{
	init();
}

CTaskGroup::~CTaskGroup()
{
	SG_UNREF(m_tasks);
}

void CTaskGroup::init()
{
	m_tasks = new CDynamicObjectArray(true);
}

void CTaskGroup::append_task(CTask* task)
{
	m_tasks->append_element(task);
}

int32_t CTaskGroup::get_num_tasks() const
{
	return m_tasks->get_num_elements();
}

SGVector<index_t>* CTaskGroup::get_tasks_indices() const
{
	int32_t n_tasks = m_tasks->get_num_elements();
	SG_DEBUG("Number of tasks = %d\n", n_tasks)

	SGVector<index_t>* tasks_indices = SG_MALLOC(SGVector<index_t>, n_tasks);
	for (int32_t i=0; i<n_tasks; i++)
	{
		CTask* task = (CTask*)m_tasks->get_element(i);
		tasks_indices[i] = task->get_indices();
		SG_UNREF(task);
	}

	return tasks_indices;
}
