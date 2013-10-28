/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/evaluation/ContingencyTableEvaluation.h>
#include <shogun/labels/BinaryLabels.h>

using namespace shogun;

float64_t CContingencyTableEvaluation::evaluate(CLabels* predicted, CLabels* ground_truth)
{
	ASSERT(predicted->get_label_type()==LT_BINARY)
	ASSERT(ground_truth->get_label_type()==LT_BINARY)

	/* commented out: what if a machine only returns +1 in apply() ??
	 * Heiko Strathamn */
//	predicted->ensure_valid();

	ground_truth->ensure_valid();
	compute_scores((CBinaryLabels*)predicted,(CBinaryLabels*)ground_truth);
	switch (m_type)
	{
		case ACCURACY:
			return get_accuracy();
		case ERROR_RATE:
			return get_error_rate();
		case BAL:
			return get_BAL();
		case WRACC:
			return get_WRACC();
		case F1:
			return get_F1();
		case CROSS_CORRELATION:
			return get_cross_correlation();
		case RECALL:
			return get_recall();
		case PRECISION:
			return get_precision();
		case SPECIFICITY:
			return get_specificity();
		case CUSTOM:
			return get_custom_score();
	}

	SG_NOTIMPLEMENTED
	return 42;
}

inline EEvaluationDirection CContingencyTableEvaluation::get_evaluation_direction() const
{
	switch (m_type)
	{
	case ACCURACY:
		return ED_MAXIMIZE;
	case ERROR_RATE:
		return ED_MINIMIZE;
	case BAL:
		return ED_MINIMIZE;
	case WRACC:
		return ED_MAXIMIZE;
	case F1:
		return ED_MAXIMIZE;
	case CROSS_CORRELATION:
		return ED_MAXIMIZE;
	case RECALL:
		return ED_MAXIMIZE;
	case PRECISION:
		return ED_MAXIMIZE;
	case SPECIFICITY:
		return ED_MAXIMIZE;
	case CUSTOM:
		return get_custom_direction();
	default:
		SG_NOTIMPLEMENTED
	}

	return ED_MINIMIZE;
}

void CContingencyTableEvaluation::compute_scores(CBinaryLabels* predicted, CBinaryLabels* ground_truth)
{
	ASSERT(ground_truth->get_label_type() == LT_BINARY)
	ASSERT(predicted->get_label_type() == LT_BINARY)

	if (predicted->get_num_labels()!=ground_truth->get_num_labels())
	{
		SG_ERROR("%s::compute_scores(): Number of predicted labels (%d) is not "
				"equal to number of ground truth labels (%d)!\n", get_name(),
				predicted->get_num_labels(), ground_truth->get_num_labels());
	}
	m_TP = 0.0;
	m_FP = 0.0;
	m_TN = 0.0;
	m_FN = 0.0;
	m_N = predicted->get_num_labels();

	for (int i=0; i<predicted->get_num_labels(); i++)
	{
		if (ground_truth->get_label(i)==1)
		{
			if (predicted->get_label(i)==1)
				m_TP += 1.0;
			else
				m_FN += 1.0;
		}
		else
		{
			if (predicted->get_label(i)==1)
				m_FP += 1.0;
			else
				m_TN += 1.0;
		}
	}
	m_computed = true;
}
