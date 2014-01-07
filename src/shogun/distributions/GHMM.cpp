/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <distributions/GHMM.h>

using namespace shogun;

CGHMM::CGHMM()
: CDistribution()
{
}

CGHMM::~CGHMM()
{
}

bool CGHMM::train(CFeatures* data)
{
	return false;
}

int32_t CGHMM::get_num_model_parameters()
{
	return 0;
}

float64_t CGHMM::get_log_model_parameter(int32_t param_num)
{
	return 0;
}

float64_t CGHMM::get_log_derivative(int32_t param_num, int32_t num_example)
{
	return 0;
}

float64_t CGHMM::get_log_likelihood_example(int32_t num_example)
{
	return 0;
}

