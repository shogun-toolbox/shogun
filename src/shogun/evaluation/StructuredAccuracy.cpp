/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012-2013 Fernando José Iglesias García
 * Copyright (C) 2012-2013 Fernando José Iglesias García
 */

#include <shogun/evaluation/StructuredAccuracy.h>
#include <shogun/structure/SequenceLabels.h>
#include <shogun/structure/MulticlassSOLabels.h>

using namespace shogun;

CStructuredAccuracy::CStructuredAccuracy() : CEvaluation()
{
}

CStructuredAccuracy::~CStructuredAccuracy()
{
}

float64_t CStructuredAccuracy::evaluate(CLabels* predicted, CLabels* ground_truth)
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

	CStructuredLabels* pred_labs = CLabelsFactory::to_structured(predicted);
	CStructuredLabels* true_labs = CLabelsFactory::to_structured(ground_truth);

	REQUIRE(pred_labs->get_structured_data_type() ==
			true_labs->get_structured_data_type(), "Predicted and ground truth "
			"labels must be composed of the same structured data\n");

	switch ( pred_labs->get_structured_data_type() )
	{
		case (SDT_REAL):
			return evaluate_real(pred_labs, true_labs);
		case (SDT_SEQUENCE):
			return evaluate_sequence(pred_labs, true_labs);
		default:
			SG_ERROR("Unknown structured data type for evaluation\n")
	}

	return 0.0;
}

SGMatrix< int32_t > CStructuredAccuracy::get_confusion_matrix(
		CLabels* predicted, CLabels* ground_truth)
{
	SG_SERROR("Not implemented\n")
	return SGMatrix< int32_t >();
}

float64_t CStructuredAccuracy::evaluate_real(CStructuredLabels* predicted,
		CStructuredLabels* ground_truth)
{
	int32_t length = predicted->get_num_labels();
	int32_t num_equal = 0;

	for ( int32_t i = 0 ; i < length ; ++i )
	{
		CRealNumber* truth =
			CRealNumber::obtain_from_generic(ground_truth->get_label(i));
		CRealNumber* pred =
			CRealNumber::obtain_from_generic(predicted->get_label(i));

		num_equal += truth->value == pred->value;

		SG_UNREF(truth);
		SG_UNREF(pred);
	}

	return (1.0*num_equal) / length;
}

float64_t CStructuredAccuracy::evaluate_sequence(CStructuredLabels* predicted,
		CStructuredLabels* ground_truth)
{
	int32_t length = predicted->get_num_labels();
	// Accuracy of each each label
	SGVector< float64_t > accuracies(length);
	int32_t num_equal = 0;

	for ( int32_t i = 0 ; i < length ; ++i )
	{
		CSequence* true_seq =
			CSequence::obtain_from_generic(ground_truth->get_label(i));
		CSequence* pred_seq =
			CSequence::obtain_from_generic(predicted->get_label(i));

		SGVector<int32_t> true_seq_data = true_seq->get_data();
		SGVector<int32_t> pred_seq_data = pred_seq->get_data();

		REQUIRE(true_seq_data.size() == pred_seq_data.size(), "Corresponding ground "
				"truth and predicted sequences must be equally long\n");

		num_equal = 0;
		// Count the number of elements that are equal in both sequences
		for ( int32_t j = 0 ; j < true_seq_data.size() ; ++j )
			num_equal += true_seq_data[j] == pred_seq_data[j];

		accuracies[i] = (1.0*num_equal) / true_seq_data.size();

		SG_UNREF(true_seq);
		SG_UNREF(pred_seq);
	}

	return accuracies.mean();
}
