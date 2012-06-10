/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Jacob Walker
 */

#include <shogun/regression/gp/ZeroMean.h>

namespace shogun {

CZeroMean::CZeroMean() {
	// TODO Auto-generated constructor stub

}

CZeroMean::~CZeroMean() {
	// TODO Auto-generated destructor stub
}

SGVector<float64_t> CZeroMean::get_mean_vector(SGMatrix<float64_t> data)
{
	SGVector<float64_t> result(data.num_rows);
	for(int i = 0; i < result.vlen; i++) result[i] = 0;
	return result;
}

}
