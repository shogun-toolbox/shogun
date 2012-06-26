/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#include <shogun/transfer/multitask/Task.h>

using namespace shogun;

CTask::CTask() : CSGObject(),
	m_task_name("task"),
	m_min_index(0), m_max_index(0),
	m_weight(1.0), m_subtasks(NULL)
{
	m_subtasks = new CList();
}

CTask::CTask(index_t min_index, index_t max_index, 
             float64_t weight, const char* name) :
	CSGObject(), m_task_name(name),
	m_min_index(min_index), m_max_index(max_index),
	m_weight(weight), m_subtasks(NULL)
{
	m_subtasks = new CList();
}

CTask::~CTask()
{
	SG_UNREF(m_subtasks);
}

void CTask::add_subtask(CTask* subtask)
{
	ASSERT(subtask->get_min_index()>=m_min_index);
	ASSERT(subtask->get_max_index()<=m_max_index);
	m_subtasks->append_element(subtask);
}

CList* CTask::get_subtasks()
{
	return m_subtasks;
}
