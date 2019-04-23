/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Soeren Sonnenburg, Bjoern Esser
 */

#include <shogun/transfer/multitask/Task.h>

using namespace shogun;

Task::Task() : SGObject()
{
	init();

	m_weight = 0.0;
	m_name = "task";
}

Task::Task(index_t min_index, index_t max_index,
             float64_t weight, const char* name) :
	SGObject()
{
	init();

	require(min_index<max_index, "min index should be less than max index");
	m_indices = SGVector<index_t>(max_index-min_index);
	for (int32_t i=0; i<m_indices.vlen; i++)
		m_indices[i] = i+min_index;
	m_weight = weight;
	m_name = name;
}

Task::Task(SGVector<index_t> indices,
             float64_t weight, const char* name) :
	SGObject()
{
	init();

	m_indices = indices;
}

void Task::init()
{
	m_subtasks = std::make_shared<List>(true);


	SG_ADD((std::shared_ptr<SGObject>*)&m_subtasks,"subtasks","subtasks of given task");
	SG_ADD(&m_indices,"indices","indices of task");
	SG_ADD(&m_weight,"weight","weight of task");
}

Task::~Task()
{

}

bool Task::is_contiguous()
{
	require(m_indices.vlen>1,"Task indices vector must not be empty or contain only one element");
	bool result = true;
	for (int32_t i=0; i<m_indices.vlen-1; i++)
	{
		if (m_indices[i]!=m_indices[i+1]-1)
		{
			result = false;
			break;
		}
	}

	return result;
}

void Task::add_subtask(std::shared_ptr<Task> subtask)
{
	SGVector<index_t> subtask_indices = subtask->get_indices();
	for (int32_t i=0; i<subtask_indices.vlen; i++)
	{
		bool found = false;
		for (int32_t j=0; j<m_indices.vlen; j++)
		{
			if (subtask_indices[i] == m_indices[j])
			{
				found = true;
				break;
			}
		}
		if (!found)
			error("Subtask contains indices that are not contained in this task");
	}
	m_subtasks->append_element(subtask);
}

std::shared_ptr<List> Task::get_subtasks()
{

	return m_subtasks;
}

int32_t Task::get_num_subtasks()
{
	return m_subtasks->get_num_elements();
}
