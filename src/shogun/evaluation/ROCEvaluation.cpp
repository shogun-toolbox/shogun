/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/evaluation/ROCEvaluation.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

CROCEvaluation::~CROCEvaluation()
{
	SG_FREE(m_ROC_graph);
}

float64_t CROCEvaluation::evaluate(CLabels* predicted, CLabels* ground_truth)
{
	ASSERT(predicted && ground_truth);
	ASSERT(predicted->get_num_labels()==ground_truth->get_num_labels());
	ASSERT(ground_truth->is_two_class_labeling());

	// assume threshold as negative infinity
	float64_t threshold = CMath::ALMOST_NEG_INFTY;
	// false positive rate
	float64_t fp = 0.0;
	// true positive rate
	float64_t tp=0.0;

	int32_t i;
	// total number of positive labels in predicted
	int32_t pos_count=0;
	int32_t neg_count=0;

	// initialize number of labels and labels
	SGVector<float64_t> orig_labels = predicted->get_labels();
	int32_t length = orig_labels.vlen;
	float64_t* labels = CMath::clone_vector(orig_labels.vector, length);
	orig_labels.free_vector();

	// get sorted indexes
	int32_t* idxs = SG_MALLOC(int32_t, length);
	for(i=0; i<length; i++)
		idxs[i] = i;

	CMath::qsort_backward_index(labels,idxs,length);

	// number of different predicted labels
	int32_t diff_count=1;

	// get number of different labels
	for (i=0; i<length-1; i++)
	{
		if (labels[i] != labels[i+1])
			diff_count++;
	}

	delete [] labels;

	// initialize graph and auROC
	SG_FREE(m_ROC_graph);
	m_ROC_graph = SG_MALLOC(float64_t, diff_count*2+2);
	m_auROC = 0.0;

	// get total numbers of positive and negative labels
	for(i=0; i<length; i++)
	{
		if (ground_truth->get_label(i) > 0)
			pos_count++;
		else
			neg_count++;
	}

	// assure both number of positive and negative examples is >0
	ASSERT(pos_count>0 && neg_count>0);

	int32_t j = 0;
	float64_t label;

	// create ROC curve and calculate auROC
	for(i=0; i<length; i++)
	{
		label = predicted->get_label(idxs[i]);

		if (label != threshold)
		{
			threshold = label;
			m_ROC_graph[2*j] = fp/neg_count;
			m_ROC_graph[2*j+1] = tp/pos_count;
			j++;
		}

		if (ground_truth->get_label(idxs[i]) > 0)
			tp+=1.0;
		else
			fp+=1.0;
	}

	// add (1,1) to ROC curve
	m_ROC_graph[2*diff_count] = 1.0;
	m_ROC_graph[2*diff_count+1] = 1.0;

	// set ROC length
	m_ROC_length = diff_count+1;

	// calc auROC using area under curve
	m_auROC = CMath::area_under_curve(m_ROC_graph,m_ROC_length,false);

	m_computed = true;

	return m_auROC;
}

SGMatrix<float64_t> CROCEvaluation::get_ROC()
{
	if (!m_computed)
		SG_ERROR("Uninitialized, please call evaluate first");

	ASSERT(m_ROC_graph);

	return SGMatrix<float64_t>(m_ROC_graph,2,m_ROC_length);
}

float64_t CROCEvaluation::get_auROC()
{
	if (!m_computed)
			SG_ERROR("Uninitialized, please call evaluate first");

	return m_auROC;
}
