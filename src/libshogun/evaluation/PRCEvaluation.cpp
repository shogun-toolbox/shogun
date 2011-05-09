/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include "evaluation/PRCEvaluation.h"
#include "lib/Mathematics.h"

using namespace shogun;

CPRCEvaluation::~CPRCEvaluation()
{
	delete[] m_PRC_graph;
}

float64_t CPRCEvaluation::evaluate(CLabels* predicted, CLabels* ground_truth)
{
	ASSERT(predicted && ground_truth);
	ASSERT(predicted->get_num_labels()==ground_truth->get_num_labels());
	ASSERT(ground_truth->is_two_class_labeling());

	// number of true positive examples
	float64_t tp = 0.0;
	int32_t i;

	// total number of positive labels in predicted
	int32_t pos_count=0;

	// initialize number of labels and labels
	int32_t length = predicted->get_num_labels();
	float64_t* labels = predicted->get_labels(length);

	// get indexes for sort
	int32_t* idxs = new int32_t[length];
	for(i=0; i<length; i++)
		idxs[i] = i;

	// sort indexes by labels ascending
	CMath::qsort_backward_index(labels,idxs,length);

	// clean and initialize graph and auPRC
	delete[] labels;
	delete[] m_PRC_graph;
	m_PRC_graph = new float64_t[length*2];
	m_auPRC = 0.0;

	// get total numbers of positive and negative labels
	for (i=0; i<length; i++)
	{
		if (ground_truth->get_label(i) > 0)
			pos_count++;
	}

	// assure number of positive examples is >0
	ASSERT(pos_count>0);

	// create PRC curve
	for (i=0; i<length; i++)
	{
		// update number of true positive examples
		if (ground_truth->get_label(idxs[i]) > 0)
			tp += 1.0;

		// precision (x)
		m_PRC_graph[i] = tp/(i+1);
		// recall (y)
		m_PRC_graph[length+i] = tp/pos_count;
	}

	// calc auRPC using area under curve
	m_auPRC = CMath::area_under_curve(m_PRC_graph+length,length,m_PRC_graph,length);

	// set PRC length and computed indicator
	m_PRC_length = length;
	m_computed = true;

	return m_auPRC;
}

void CPRCEvaluation::get_PRC(float64_t** result, int32_t* num, int32_t* dim)
{
	if (!m_computed)
		SG_ERROR("Uninitialized, please call evaluate first");

	ASSERT(m_PRC_graph);
	*num = m_PRC_length;
	*dim = 2;

	*result = (float64_t*) SG_MALLOC(sizeof(float64_t)*m_PRC_length*2);
	memcpy(*result, m_PRC_graph, m_PRC_length*2*sizeof(float64_t));
}

float64_t CPRCEvaluation::get_auPRC()
{
	if (!m_computed)
			SG_ERROR("Uninitialized, please call evaluate first");

	return m_auPRC;
}


