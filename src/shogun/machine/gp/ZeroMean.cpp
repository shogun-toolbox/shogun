/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Roman Votyakov
 * Copyright (C) 2012 Jacob Walker
 */

#include <machine/gp/ZeroMean.h>

using namespace shogun;

CZeroMean::CZeroMean() : CMeanFunction()
{
}

CZeroMean::~CZeroMean()
{
}

SGVector<float64_t> CZeroMean::get_mean_vector(const CFeatures* features) const
{
	SGVector<float64_t> result(features->get_num_vectors());
	result.zero();
	return result;
}
