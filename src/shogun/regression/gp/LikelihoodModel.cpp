/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Jacob Walker
 */

#include <shogun/regression/gp/LikelihoodModel.h>
#ifdef HAVE_EIGEN3

using namespace shogun;

CLikelihoodModel::CLikelihoodModel()
{
}

CLikelihoodModel::~CLikelihoodModel()
{
}

float64_t CLikelihoodModel::get_parameter_derivative(const char* param_name)
{
	SG_ERROR("Derivative with respect to parameter %s " \
			"not implemented in likelihood model (%s).",
			param_name, get_name());

	return 0;
}
#endif /* HAVE_EIGEN3 */
