/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Roman Votyakov
 * Written (W) 2013 Heiko Strathmann
 * Copyright (C) 2012 Jacob Walker
 * Copyright (C) 2013 Roman Votyakov
 */

#include <shogun/machine/gp/LikelihoodModel.h>

using namespace shogun;

CLikelihoodModel::CLikelihoodModel()
{
}

CLikelihoodModel::~CLikelihoodModel()
{
}

SGVector<float64_t> CLikelihoodModel::get_log_probability_f(CLabels* lab,
		SGMatrix<float64_t> F)
{
	REQUIRE(lab, "Given labels are NULL!\n");
	REQUIRE(lab->get_num_labels()==F.num_rows, "Number of labels (%d) does "
			"not match dimension of functions (%d)\n",
			lab->get_num_labels(),F.num_rows);
	REQUIRE(F.num_cols>0, "Number of passed functions (%d) must be positive\n",
			F.num_cols);

	SGVector<float64_t> result(F.num_cols);
	for (index_t i=0; i<F.num_cols; ++i)
	{
		/* extract current sample from matrix, assume col-major, dont copy */
		SGVector<float64_t> f(&F.matrix[i*F.num_rows], F.num_rows, false);
		result[i]=SGVector<float64_t>::sum(get_log_probability_f(lab, f));
	}

	return result;
}
