/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Viktor Gal
 * Copyright (C) 2013 Viktor Gal
 */

#include <shogun/ensemble/MeanRule.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/linalg/linalg.h>

using namespace shogun;

CMeanRule::CMeanRule()
	: CCombinationRule()
{

}

CMeanRule::~CMeanRule()
{

}

SGVector<float64_t> CMeanRule::combine(const SGMatrix<float64_t>& ensemble_result) const
{
	float64_t* row_sum =
		SGMatrix<float64_t>::get_column_sum(ensemble_result.matrix,
											ensemble_result.num_rows,
											ensemble_result.num_cols);

	SGVector<float64_t> mean_labels(row_sum, ensemble_result.num_rows);

	float64_t scale = 1/(float64_t)ensemble_result.num_cols;
	linalg::scale<linalg::Backend::NATIVE>(mean_labels, scale);

	return mean_labels;
}

float64_t CMeanRule::combine(const SGVector<float64_t>& ensemble_result) const
{
	float64_t combined = SGVector<float64_t>::sum(ensemble_result);
	combined /= (float64_t)ensemble_result.vlen;
	return combined;
}
