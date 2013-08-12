/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Jacob Walker
 */

#include <shogun/machine/gp/ZeroMean.h>

using namespace shogun;

CZeroMean::CZeroMean() : CMeanFunction()
{
}

CZeroMean::~CZeroMean()
{
}

SGVector<float64_t> CZeroMean::get_mean_vector(SGMatrix<float64_t> data) const
{
	SGVector<float64_t> result(data.num_cols);
	result.set_const(0.0);
	return result;
}
