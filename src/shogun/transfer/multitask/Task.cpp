/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#include <transfer/multitask/Task.h>

using namespace shogun;

CTask::CTask() : CSGObject()
{
	init();

	m_weight = 0.0;
	m_name = "task";
}

CTask::CTask(index_t min_index, index_t max_index,
             float64_t weight, const char* name) :
	CSGObject()
{
	init();

	REQUIRE(min_index<max_index, "min index should be less than max index")
	m_indices = SGVector<index_t>(max_index-min_index);
	for (int32_t i=0; i<m_indices.vlen; i++)
		m_indices[i] = i+min_index;
	m_weight = weight;
	m_name = name;
}

CTask::CTask(SGVector<index_t> indices,
             float64_t weight, const char* name) :
	CSGObject()
{
	init();

	m_indices = indices;
}

void CTask::init()
{
	m_subtasks = new CList(true);
	SG_REF(m_subtasks);

	SG_ADD((CSGObject**)&m_subtasks,"subtasks","subtasks of given task", MS_NOT_AVAILABLE);
	SG_ADD(&m_indices,"indices","indices of task", MS_NOT_AVAILABLE);
	SG_ADD(&m_weight,"weight","weight of task", MS_NOT_AVAILABLE);
}

CTask::~CTask()
{
	SG_UNREF(m_subtasks);
}

bool CTask::is_contiguous()
{
	REQUIRE(m_indices.vlen>1,"Task indices vector must not be empty or contain only one element")
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

void CTask::add_subtask(CTask* subtask)
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
			SG_ERROR("Subtask contains indices that are not contained in this task\n")
	}
	m_subtasks->append_element(subtask);
}

CList* CTask::get_subtasks()
{
	SG_REF(m_subtasks);
	return m_subtasks;
}

int32_t CTask::get_num_subtasks()
{
	return m_subtasks->get_num_elements();
}
