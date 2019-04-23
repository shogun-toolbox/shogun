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

StructuredAccuracy::StructuredAccuracy() : Evaluation()
{
}

StructuredAccuracy::~StructuredAccuracy()
{
}

float64_t StructuredAccuracy::evaluate(std::shared_ptr<Labels > predicted, std::shared_ptr<Labels > ground_truth)
{
	require(predicted && ground_truth, "Labels objects passed to evaluate "
	        "cannot be null");
	require(predicted->get_num_labels() == ground_truth->get_num_labels(),
	        "The number of predicted and ground truth labels must "
	        "be the same");
	require(predicted->get_label_type() == LT_STRUCTURED, "The predicted "
	        "labels must be of type StructuredLabels");
	require(ground_truth->get_label_type() == LT_STRUCTURED, "The ground truth "
	        "labels must be of type StructuredLabels");

	auto pred_labs = std::dynamic_pointer_cast<StructuredLabels>(predicted);
	auto true_labs = std::dynamic_pointer_cast<StructuredLabels>(ground_truth);

	require(pred_labs->get_structured_data_type() ==
	        true_labs->get_structured_data_type(), "Predicted and ground truth "
	        "labels must be composed of the same structured data");

	switch (pred_labs->get_structured_data_type())
	{
	case (SDT_REAL):
		return evaluate_real(pred_labs, true_labs);

	case (SDT_SEQUENCE):
		return evaluate_sequence(pred_labs, true_labs);

	case (SDT_SPARSE_MULTILABEL):
		return evaluate_sparse_multilabel(pred_labs, true_labs);

	default:
		error("Unknown structured data type for evaluation");
	}

	return 0.0;
}

SGMatrix<int32_t> StructuredAccuracy::get_confusion_matrix(
        std::shared_ptr<Labels > predicted, std::shared_ptr<Labels > ground_truth)
{
	error("Not implemented");
	return SGMatrix<int32_t>();
}

float64_t StructuredAccuracy::evaluate_real(std::shared_ptr<StructuredLabels > predicted,
                std::shared_ptr<StructuredLabels > ground_truth)
{
	int32_t length = predicted->get_num_labels();
	int32_t num_equal = 0;

	for (int32_t i = 0 ; i < length ; ++i)
	{
		auto truth = std::dynamic_pointer_cast<RealNumber>(ground_truth->get_label(i));
		auto pred = std::dynamic_pointer_cast<RealNumber>(predicted->get_label(i));

		num_equal += truth->value == pred->value;



	}

	return (1.0 * num_equal) / length;
}

float64_t StructuredAccuracy::evaluate_sequence(std::shared_ptr<StructuredLabels > predicted,
                std::shared_ptr<StructuredLabels > ground_truth)
{
	int32_t length = predicted->get_num_labels();
	// Accuracy of each each label
	SGVector<float64_t> accuracies(length);
	int32_t num_equal = 0;

	for (int32_t i = 0 ; i < length ; ++i)
	{
		auto true_seq = std::dynamic_pointer_cast<Sequence>(ground_truth->get_label(i));
		auto pred_seq = std::dynamic_pointer_cast<Sequence>(predicted->get_label(i));

		SGVector<int32_t> true_seq_data = true_seq->get_data();
		SGVector<int32_t> pred_seq_data = pred_seq->get_data();

		require(true_seq_data.size() == pred_seq_data.size(), "Corresponding ground "
		        "truth and predicted sequences must be equally long");

		num_equal = 0;

		// Count the number of elements that are equal in both sequences
		for (int32_t j = 0 ; j < true_seq_data.size() ; ++j)
		{
			num_equal += true_seq_data[j] == pred_seq_data[j];
		}

		accuracies[i] = (1.0 * num_equal) / true_seq_data.size();



	}

	return Statistics::mean(accuracies);
}

float64_t StructuredAccuracy::evaluate_sparse_multilabel(std::shared_ptr<StructuredLabels > predicted,
                std::shared_ptr<StructuredLabels > ground_truth)
{
	auto multi_pred = std::static_pointer_cast<MultilabelSOLabels>(predicted);
	auto multi_truth = std::static_pointer_cast<MultilabelSOLabels>(ground_truth);

	auto evaluator = std::make_shared<MultilabelAccuracy>();


	float64_t accuracy = evaluator->evaluate(multi_pred->get_multilabel_labels(),
	                     multi_truth->get_multilabel_labels());



	return accuracy;
}

