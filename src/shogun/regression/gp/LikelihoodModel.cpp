/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Jacob Walker
 */

#include <shogun/regression/gp/LikelihoodModel.h>

using namespace shogun;

CLikelihoodModel::CLikelihoodModel() {
	// TODO Auto-generated constructor stub

}

CLikelihoodModel::~CLikelihoodModel() {
	// TODO Auto-generated destructor stub
}

float64_t CLikelihoodModel::get_parameter_derivative(const char* param_name)
{
	SG_ERROR("Derivative with respect to parameter %s not implemented in", param_name);
	SG_ERROR("likelihood model (%s). Returning zero.", get_name());
	return 0;
}
