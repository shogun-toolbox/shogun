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

CTask::CTask() : CIndexBlock()
{
}

CTask::CTask(index_t min_index, index_t max_index, 
             float64_t weight, const char* name) :
	CIndexBlock(min_index, max_index, weight, name)
{
}

CTask::~CTask()
{
}
