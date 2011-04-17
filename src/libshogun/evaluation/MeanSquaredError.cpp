/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include "MeanSquaredError.h"
#include "features/Labels.h"
#include "lib/Mathematics.h"

using namespace shogun;

float64_t CMeanSquaredError::evaluate(CLabels* labels, CLabels* labels_valid)
{
	ASSERT(labels->get_num_labels() == labels_valid->get_num_labels());
	int32_t length = labels->get_num_labels();
	float64_t mse = 0.0;
	for (int32_t i=0; i<length; i++)
		mse += CMath::sq(labels->get_label(i) - labels_valid->get_label(i));
	mse /= length;
	return mse;
}
