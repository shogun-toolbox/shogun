/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/evaluation/PRCEvaluation.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

CPRCEvaluation::~CPRCEvaluation()
{
	SG_FREE(m_PRC_graph);
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
	SGVector<float64_t> orig_labels = predicted->get_labels();
	int32_t length = orig_labels.vlen;
	float64_t* labels = CMath::clone_vector(orig_labels.vector, length);
	orig_labels.free_vector();

	// get indexes for sort
	int32_t* idxs = SG_MALLOC(int32_t, length);
	for(i=0; i<length; i++)
		idxs[i] = i;

	// sort indexes by labels ascending
	CMath::qsort_backward_index(labels,idxs,length);

	// clean and initialize graph and auPRC
	SG_FREE(labels);
	SG_FREE(m_PRC_graph);
	m_PRC_graph = SG_MALLOC(float64_t, length*2);
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
		m_PRC_graph[2*i] = tp/float64_t(i+1);
		// recall (y)
		m_PRC_graph[2*i+1] = tp/float64_t(pos_count);
	}

	// calc auRPC using area under curve
	m_auPRC = CMath::area_under_curve(m_PRC_graph,length,true);

	// set PRC length and computed indicator
	m_PRC_length = length;
	m_computed = true;

	return m_auPRC;
}

SGMatrix<float64_t> CPRCEvaluation::get_PRC()
{
	if (!m_computed)
		SG_ERROR("Uninitialized, please call evaluate first");

	ASSERT(m_PRC_graph);

	return SGMatrix<float64_t>(m_PRC_graph,2,m_PRC_length);
}

float64_t CPRCEvaluation::get_auPRC()
{
	if (!m_computed)
			SG_ERROR("Uninitialized, please call evaluate first");

	return m_auPRC;
}


