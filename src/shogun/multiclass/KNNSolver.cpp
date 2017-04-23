/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */
 
#include <shogun/labels/Labels.h>
#include <shogun/lib/Time.h>
#include <shogun/base/Parameter.h>
#include <shogun/lib/Signal.h>
#include <shogun/multiclass/KNNSolver.h>

using namespace shogun;

CKNNSolver::CKNNSolver(const int32_t k, const float64_t q, const int32_t num_classes, const int32_t min_label, const SGVector<int32_t> train_labels)
: CDistanceMachine()
{
	init();

	m_k=k;
	m_q=q;
	m_num_classes=num_classes;
	m_min_label=min_label;
	m_train_labels=train_labels;
}

void CKNNSolver::init()
{
	m_k=3;
	m_q=1.0;
	m_num_classes=0;
	m_min_label=0;
	m_train_labels=0;
}

int32_t CKNNSolver::choose_class(float64_t* classes, const int32_t* train_lab) const
{
	memset(classes, 0, sizeof(float64_t)*m_num_classes);

	float64_t multiplier = m_q;
	for (int32_t j=0; j<m_k; j++)
	{
		classes[train_lab[j]]+= multiplier;
		multiplier*= multiplier;
	}

	//choose the class that got 'outputted' most often
	int32_t out_idx=0;
	float64_t out_max=0;

	for (index_t j=0; j<m_num_classes; j++)
	{
		if (out_max< classes[j])
		{
			out_idx= j;
			out_max= classes[j];
		}
	}

	return out_idx;
}

void CKNNSolver::choose_class_for_multiple_k(int32_t* output, int32_t* classes, const int32_t* train_lab, const int32_t step) const
{
	//compute histogram of class outputs of the first k nearest neighbours
	memset(classes, 0, sizeof(int32_t)*m_num_classes);

	for (index_t j=0; j<m_k; j++)
	{
		classes[train_lab[j]]++;

		//choose the class that got 'outputted' most often
		int32_t out_idx=0;
		int32_t out_max=0;

		for (index_t c=0; c<m_num_classes; c++)
		{
			if (out_max< classes[c])
			{
				out_idx= c;
				out_max= classes[c];
			}
		}

		output[j*step]=out_idx+m_min_label;
	}
}
