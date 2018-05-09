/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Fernando Iglesias, Evgeniy Andreev, Sergey Lisitsyn, 
 *          Soeren Sonnenburg, Sanuj Sharma, Abinash Panda
 */

#include <shogun/evaluation/StructuredAccuracy.h>
#include <shogun/structure/SequenceLabels.h>
#include <shogun/structure/MulticlassSOLabels.h>
#include <shogun/structure/MultilabelSOLabels.h>
#include <shogun/evaluation/MultilabelAccuracy.h>
#include <shogun/mathematics/Statistics.h>

using namespace shogun;

CStructuredAccuracy::CStructuredAccuracy() : CEvaluation()
{
}

CStructuredAccuracy::~CStructuredAccuracy()
{
}

float64_t CStructuredAccuracy::evaluate(CLabels * predicted, CLabels * ground_truth)
{
	REQUIRE(predicted && ground_truth, "CLabels objects passed to evaluate "
	        "cannot be null\n");
	REQUIRE(predicted->get_num_labels() == ground_truth->get_num_labels(),
	        "The number of predicted and ground truth labels must "
	        "be the same\n");
	REQUIRE(predicted->get_label_type() == LT_STRUCTURED, "The predicted "
	        "labels must be of type CStructuredLabels\n");
	REQUIRE(ground_truth->get_label_type() == LT_STRUCTURED, "The ground truth "
	        "labels must be of type CStructuredLabels\n");

	CStructuredLabels * pred_labs = predicted->as<CStructuredLabels>();
	CStructuredLabels * true_labs = ground_truth->as<CStructuredLabels>();

	REQUIRE(pred_labs->get_structured_data_type() ==
	        true_labs->get_structured_data_type(), "Predicted and ground truth "
	        "labels must be composed of the same structured data\n");

	switch (pred_labs->get_structured_data_type())
	{
	case (SDT_REAL):
		return evaluate_real(pred_labs, true_labs);

	case (SDT_SEQUENCE):
		return evaluate_sequence(pred_labs, true_labs);

	case (SDT_SPARSE_MULTILABEL):
		return evaluate_sparse_multilabel(pred_labs, true_labs);

	default:
		SG_ERROR("Unknown structured data type for evaluation\n")
	}

	return 0.0;
}

SGMatrix<int32_t> CStructuredAccuracy::get_confusion_matrix(
        CLabels * predicted, CLabels * ground_truth)
{
	SG_SERROR("Not implemented\n")
	return SGMatrix<int32_t>();
}

float64_t CStructuredAccuracy::evaluate_real(CStructuredLabels * predicted,
                CStructuredLabels * ground_truth)
{
	int32_t length = predicted->get_num_labels();
	int32_t num_equal = 0;

	for (int32_t i = 0 ; i < length ; ++i)
	{
		CRealNumber * truth = ground_truth->get_label(i)->as<CRealNumber>();
		CRealNumber * pred = predicted->get_label(i)->as<CRealNumber>();

		num_equal += truth->value == pred->value;

		SG_UNREF(truth);
		SG_UNREF(pred);
	}

	return (1.0 * num_equal) / length;
}

float64_t CStructuredAccuracy::evaluate_sequence(CStructuredLabels * predicted,
                CStructuredLabels * ground_truth)
{
	int32_t length = predicted->get_num_labels();
	// Accuracy of each each label
	SGVector<float64_t> accuracies(length);
	int32_t num_equal = 0;

	for (int32_t i = 0 ; i < length ; ++i)
	{
		CSequence * true_seq = ground_truth->get_label(i)->as<CSequence>();
		CSequence * pred_seq = predicted->get_label(i)->as<CSequence>();

		SGVector<int32_t> true_seq_data = true_seq->get_data();
		SGVector<int32_t> pred_seq_data = pred_seq->get_data();

		REQUIRE(true_seq_data.size() == pred_seq_data.size(), "Corresponding ground "
		        "truth and predicted sequences must be equally long\n");

		num_equal = 0;

		// Count the number of elements that are equal in both sequences
		for (int32_t j = 0 ; j < true_seq_data.size() ; ++j)
		{
			num_equal += true_seq_data[j] == pred_seq_data[j];
		}

		accuracies[i] = (1.0 * num_equal) / true_seq_data.size();

		SG_UNREF(true_seq);
		SG_UNREF(pred_seq);
	}

	return CStatistics::mean(accuracies);
}

float64_t CStructuredAccuracy::evaluate_sparse_multilabel(CStructuredLabels * predicted,
                CStructuredLabels * ground_truth)
{
	CMultilabelSOLabels * multi_pred = (CMultilabelSOLabels *) predicted;
	CMultilabelSOLabels * multi_truth = (CMultilabelSOLabels *) ground_truth;

	CMultilabelAccuracy * evaluator = new CMultilabelAccuracy();
	SG_REF(evaluator);

	float64_t accuracy = evaluator->evaluate(multi_pred->get_multilabel_labels(),
	                     multi_truth->get_multilabel_labels());

	SG_UNREF(evaluator);

	return accuracy;
}

