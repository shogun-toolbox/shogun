/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#include <transfer/multitask/MultitaskROCEvaluation.h>
#include <mathematics/Math.h>

#include <set>
#include <vector>

using namespace std;
using namespace shogun;

void CMultitaskROCEvaluation::set_indices(SGVector<index_t> indices)
{
	indices.display_vector("indices");
	ASSERT(m_task_relation)

	set<index_t> indices_set;
	for (int32_t i=0; i<indices.vlen; i++)
		indices_set.insert(indices[i]);

	if (m_num_tasks>0)
	{
		SG_FREE(m_tasks_indices);
	}
	m_num_tasks = m_task_relation->get_num_tasks();
	m_tasks_indices = SG_MALLOC(SGVector<index_t>, m_num_tasks);

	SGVector<index_t>* tasks_indices = m_task_relation->get_tasks_indices();
	for (int32_t t=0; t<m_num_tasks; t++)
	{
		vector<index_t> task_indices_cut;
		SGVector<index_t> task_indices = tasks_indices[t];
		//task_indices.display_vector("task indices");
		for (int32_t i=0; i<task_indices.vlen; i++)
		{
			if (indices_set.count(task_indices[i]))
			{
				//SG_SPRINT("%d is in %d task\n",task_indices[i],t)
				task_indices_cut.push_back(task_indices[i]);
			}
		}

		SGVector<index_t> cutted(task_indices_cut.size());
		for (int32_t i=0; i<cutted.vlen; i++)
			cutted[i] = task_indices_cut[i];
		//cutted.display_vector("cutted");
		m_tasks_indices[t] = cutted;
	}
	SG_FREE(tasks_indices);
}

float64_t CMultitaskROCEvaluation::evaluate(CLabels* predicted, CLabels* ground_truth)
{
	//SG_SPRINT("Evaluate\n")
	predicted->remove_all_subsets();
	ground_truth->remove_all_subsets();
	float64_t result = 0.0;
	for (int32_t t=0; t<m_num_tasks; t++)
	{
		//SG_SPRINT("%d task", t)
		//m_tasks_indices[t].display_vector();
		predicted->add_subset(m_tasks_indices[t]);
		ground_truth->add_subset(m_tasks_indices[t]);
		result += evaluate_roc(predicted,ground_truth)/m_tasks_indices[t].vlen;
		predicted->remove_subset();
		ground_truth->remove_subset();
	}
	return result;
}
