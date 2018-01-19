/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Björn Esser, Viktor Gal
 */

#include <shogun/ensemble/MeanRule.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>

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
	mean_labels.scale(scale);

	return mean_labels;
}

float64_t CMeanRule::combine(const SGVector<float64_t>& ensemble_result) const
{
	float64_t combined = SGVector<float64_t>::sum(ensemble_result);
	combined /= (float64_t)ensemble_result.vlen;
	return combined;
}
