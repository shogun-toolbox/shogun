/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sanuj Sharma, Sergey Lisitsyn, Leon Kuchenbecker
 */

#include <shogun/transfer/multitask/MultitaskROCEvaluation.h>
#include <shogun/mathematics/Math.h>

#include <set>
#include <vector>

using namespace std;
using namespace shogun;

void MultitaskROCEvaluation::set_indices(SGVector<index_t> indices)
{
	indices.display_vector("indices");
	ASSERT(m_task_relation)

	std::set<index_t> indices_set;
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
				//io::print("{} is in {} task\n",task_indices[i],t);
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

float64_t MultitaskROCEvaluation::evaluate(std::shared_ptr<Labels> predicted, std::shared_ptr<Labels> ground_truth)
{
	require(predicted->get_label_type()==LT_BINARY, "ROC evalution requires binary labels.");
	require(ground_truth->get_label_type()==LT_BINARY, "ROC evalution requires binary labels.");

        auto predicted_binary = binary_labels(predicted);
        auto ground_truth_binary = binary_labels(ground_truth);

	//io::print("Evaluate\n");
	predicted->remove_all_subsets();
	ground_truth->remove_all_subsets();
	float64_t result = 0.0;
	for (int32_t t=0; t<m_num_tasks; t++)
	{
		//io::print("{} task", t);
		//m_tasks_indices[t].display_vector();
		predicted->add_subset(m_tasks_indices[t]);
		ground_truth->add_subset(m_tasks_indices[t]);
		result += evaluate_roc(predicted_binary,ground_truth_binary)/m_tasks_indices[t].vlen;
		predicted->remove_subset();
		ground_truth->remove_subset();
	}
	return result;
}
