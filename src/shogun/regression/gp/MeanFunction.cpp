/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Jacob Walker
 */

#include <shogun/regression/gp/MeanFunction.h>

namespace shogun {

CMeanFunction::CMeanFunction() {

}

CMeanFunction::~CMeanFunction() {

}

float64_t CMeanFunction::get_parameter_derivative(SGMatrix<float64_t>& data, const char* param_name)
{
	SG_ERROR("%s has no implementation for derivative with respect to %s", get_name(), param_name);
	return 0;
}

}
