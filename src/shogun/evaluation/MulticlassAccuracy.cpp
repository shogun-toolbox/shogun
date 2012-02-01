/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/evaluation/MulticlassAccuracy.h>
#include <shogun/features/Labels.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

float64_t CMulticlassAccuracy::evaluate(CLabels* predicted, CLabels* ground_truth)
{
	ASSERT(predicted->get_num_labels() == ground_truth->get_num_labels());
	int32_t length = predicted->get_num_labels();
	float64_t accuracy = 0.0;
	for (int32_t i=0; i<length; i++)
	{
		if (predicted->get_int_label(i)==ground_truth->get_int_label(i))
			accuracy += 1.0;
	}
	accuracy /= length;
	return accuracy;
}

SGMatrix<int32_t> CMulticlassAccuracy::confusion_matrix(CLabels* predicted, CLabels* ground_truth)
{
	ASSERT(predicted->get_num_labels() == ground_truth->get_num_labels());
	int32_t length = ground_truth->get_num_labels();
	int32_t num_classes = ground_truth->get_num_classes();
	SGMatrix<int32_t> confusion_matrix(num_classes, num_classes);
	memset(confusion_matrix.matrix,0,sizeof(int32_t)*num_classes*num_classes);
	for (int32_t i=0; i<length; i++)
	{
		int32_t predicted_label = predicted->get_int_label(i);
		int32_t ground_truth_label = ground_truth->get_int_label(i);
		confusion_matrix[predicted_label*num_classes+ground_truth_label]++;
	}
	return confusion_matrix;
}

