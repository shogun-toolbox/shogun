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
#include <shogun/labels/RegressionLabels.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

CPRCEvaluation::~CPRCEvaluation()
{
}

float64_t CPRCEvaluation::evaluate(CLabels* predicted, CLabels* ground_truth)
{
	ASSERT(predicted && ground_truth)
	ASSERT(predicted->get_num_labels()==ground_truth->get_num_labels())
	ASSERT(predicted->get_label_type()==LT_BINARY)
	ASSERT(ground_truth->get_label_type()==LT_BINARY)
	ground_truth->ensure_valid();

	// number of true positive examples
	float64_t tp = 0.0;
	int32_t i;

	// total number of positive labels in predicted
	int32_t pos_count=0;

	// initialize number of labels and labels
	SGVector<float64_t> orig_labels = predicted->get_values();
	int32_t length = orig_labels.vlen;
	float64_t* labels = SGVector<float64_t>::clone_vector(orig_labels.vector, length);

	// get indexes for sort
	int32_t* idxs = SG_MALLOC(int32_t, length);
	for(i=0; i<length; i++)
		idxs[i] = i;

	// sort indexes by labels ascending
	CMath::qsort_backward_index(labels,idxs,length);

	// clean and initialize graph and auPRC
	SG_FREE(labels);
	m_PRC_graph = SGMatrix<float64_t>(2,length);
	m_thresholds = SGVector<float64_t>(length);
	m_auPRC = 0.0;

	// get total numbers of positive and negative labels
	for (i=0; i<length; i++)
	{
		if (ground_truth->get_value(i) > 0)
			pos_count++;
	}

	// assure number of positive examples is >0
	ASSERT(pos_count>0)

	// create PRC curve
	for (i=0; i<length; i++)
	{
		// update number of true positive examples
		if (ground_truth->get_value(idxs[i]) > 0)
			tp += 1.0;

		// precision (x)
		m_PRC_graph[2*i] = tp/float64_t(i+1);
		// recall (y)
		m_PRC_graph[2*i+1] = tp/float64_t(pos_count);

		m_thresholds[i]= predicted->get_value(idxs[i]);
	}

	// calc auRPC using area under curve
	m_auPRC = CMath::area_under_curve(m_PRC_graph.matrix,length,true);

	// set computed indicator
	m_computed = true;

	SG_FREE(idxs);
	return m_auPRC;
}

SGMatrix<float64_t> CPRCEvaluation::get_PRC()
{
	if (!m_computed)
		SG_ERROR("Uninitialized, please call evaluate first")

	return m_PRC_graph;
}

SGVector<float64_t> CPRCEvaluation::get_thresholds()
{
	if (!m_computed)
		SG_ERROR("Uninitialized, please call evaluate first")

	return m_thresholds;
}

float64_t CPRCEvaluation::get_auPRC()
{
	if (!m_computed)
			SG_ERROR("Uninitialized, please call evaluate first")

	return m_auPRC;
}
