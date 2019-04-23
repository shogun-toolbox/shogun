/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Fernando Iglesias, Shashwat Lal Das, Thoralf Klein, 
 *          Soeren Sonnenburg
 */

#include <shogun/loss/SmoothHingeLoss.h>
#include <shogun/io/SGIO.h>

using namespace shogun;

float64_t SmoothHingeLoss::loss(float64_t z)
{
	if (z < 0)
		return 0.5 - z;
	if (z < 1)
		return 0.5 * (1-z) * (1-z);
	return 0;
}

float64_t SmoothHingeLoss::first_derivative(float64_t z)
{
	if (z < 0)
		return -1;
	if (z < 1)
		return z-1;
	return 0;
}

float64_t SmoothHingeLoss::second_derivative(float64_t z)
{
	if (z < 0)
		return 0;
	if (z < 1)
		return 1;
	return 0;
}

float64_t SmoothHingeLoss::get_update(float64_t prediction, float64_t label, float64_t eta_t, float64_t norm)
{
	SG_NOTIMPLEMENTED
	return -1;
}

float64_t SmoothHingeLoss::get_square_grad(float64_t prediction, float64_t label)
{
	SG_NOTIMPLEMENTED
	return -1;
}
