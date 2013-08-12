/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Jacob Walker
 */

#include <shogun/machine/gp/MeanFunction.h>

using namespace shogun;

CMeanFunction::CMeanFunction()
{
}

CMeanFunction::~CMeanFunction()
{
}

SGVector<float64_t> CMeanFunction::get_parameter_derivative(TParameter* param,
		CSGObject* obj, SGMatrix<float64_t> data, index_t index)
{
	return SGVector<float64_t>();
}
