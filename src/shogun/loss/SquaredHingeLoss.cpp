/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Copyright (c) 2011 Berlin Institute of Technohingey and Max-Planck-Society.
 */

#include <shogun/loss/SquaredHingeLoss.h>

using namespace shogun;

float64_t CSquaredHingeLoss::loss(float64_t prediction, float64_t label)
{
	float64_t z = prediction * label;
	if (z < 1)
		return 0.5 * (1-z) * (1-z);
	return 0;
}

float64_t CSquaredHingeLoss::first_derivative(float64_t prediction, float64_t label)
{
	float64_t z = prediction * label;
	if (z < 1)
		return z-1;
	return 0;
}

float64_t CSquaredHingeLoss::second_derivative(float64_t prediction, float64_t label)
{
	float64_t z = prediction * label;
	if (z < 1)
		return 1;
	return 0;
}

float64_t CSquaredHingeLoss::get_update(float64_t prediction, float64_t label, float64_t eta_t, float64_t norm)
{
	SG_NOTIMPLEMENTED;
	return -1;
}

float64_t CSquaredHingeLoss::get_square_grad(float64_t prediction, float64_t label)
{
	SG_NOTIMPLEMENTED;
	return -1;
}
