/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn, Heiko Strathmann,
 *          Roman Votyakov, Viktor Gal
 */

#include <shogun/evaluation/ContingencyTableEvaluation.h>
#include <shogun/labels/BinaryLabels.h>

using namespace shogun;

ContingencyTableEvaluation::ContingencyTableEvaluation()
    : ContingencyTableEvaluation(ACCURACY)
{
}

ContingencyTableEvaluation::ContingencyTableEvaluation(
    EContingencyTableMeasureType type)
    : BinaryClassEvaluation(), m_type(type), m_computed(false)
{
	SG_ADD_OPTIONS(
	    (machine_int_t*)&m_type, "type", "type of measure to evaluate",
	    ParameterProperties::NONE,
	    SG_OPTIONS(
	        ACCURACY, ERROR_RATE, BAL, WRACC, F1, CROSS_CORRELATION, RECALL,
	        PRECISION, SPECIFICITY, CUSTOM));
}

float64_t
ContingencyTableEvaluation::evaluate(std::shared_ptr<Labels> predicted, std::shared_ptr<Labels> ground_truth)
{
	require(
	    predicted->get_num_labels() == ground_truth->get_num_labels(),
	    "Number of predicted labels ({}) must be "
	    "equal to number of ground truth labels ({})!",
	    get_name(), predicted->get_num_labels(),
	    ground_truth->get_num_labels());

	auto predicted_binary = binary_labels(predicted);
	auto ground_truth_binary = binary_labels(ground_truth);

	ground_truth->ensure_valid();
	compute_scores(predicted_binary, ground_truth_binary);
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

	not_implemented(SOURCE_LOCATION);
	return 42;
}

EEvaluationDirection
ContingencyTableEvaluation::get_evaluation_direction() const
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
		not_implemented(SOURCE_LOCATION);
	}

	return ED_MINIMIZE;
}

void ContingencyTableEvaluation::compute_scores(std::shared_ptr<BinaryLabels> predicted, std::shared_ptr<BinaryLabels> ground_truth)
{
	m_TP = 0.0;
	m_FP = 0.0;
	m_TN = 0.0;
	m_FN = 0.0;
	m_N = predicted->get_num_labels();

	for (int i = 0; i < predicted->get_num_labels(); i++)
	{
		if (ground_truth->get_label(i) == 1)
		{
			if (predicted->get_label(i) == 1)
				m_TP += 1.0;
			else
				m_FN += 1.0;
		}
		else
		{
			if (predicted->get_label(i) == 1)
				m_FP += 1.0;
			else
				m_TN += 1.0;
		}
	}
	m_computed = true;
}
