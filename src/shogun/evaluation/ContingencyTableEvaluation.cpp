/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn, Heiko Strathmann, 
 *          Roman Votyakov, Viktor Gal
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

EEvaluationDirection CContingencyTableEvaluation::get_evaluation_direction() const
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
