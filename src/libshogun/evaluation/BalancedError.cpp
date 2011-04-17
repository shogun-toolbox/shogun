/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include "BalancedError.h"
#include "features/Labels.h"
#include "lib/Mathematics.h"

using namespace shogun;

float64_t CBalancedError::evaluate(CLabels* predicted, CLabels* ground_truth)
{
	ASSERT(predicted->get_num_labels() == ground_truth->get_num_labels());
	int32_t length = predicted->get_num_labels();
	float64_t pos_err = 0.0;
	int32_t pos_count = 0.0;
	float64_t neg_err = 0.0;
	for (int i=0; i<length; i++)
	{
		if (CMath::sign(ground_truth->get_label(i)) == 1)
		{
			pos_err += predicted->get_label(i);
			pos_count++;
		}
		else
		{
			neg_err += predicted->get_label(i);
		}
	}
	pos_err /= pos_count;
	neg_err /= (length-pos_count);
	return 0.5*(pos_err+neg_err);
}
