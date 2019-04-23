/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Fernando Iglesias, Shashwat Lal Das
 */

#include <shogun/loss/HingeLoss.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

float64_t HingeLoss::loss(float64_t prediction, float64_t label)
{
	float64_t e = 1 - label * prediction;

	return (e > 0) ? e : 0;
}

float64_t HingeLoss::loss(float64_t z)
{
	return Math::max(0.0, z);
}

float64_t HingeLoss::first_derivative(float64_t prediction, float64_t label)
{
	return (label * prediction >= label * label) ? 0 : -label;
}

float64_t HingeLoss::first_derivative(float64_t z)
{
	return z > 0.0 ? 1.0 : 0.0;
}

float64_t HingeLoss::second_derivative(float64_t prediction, float64_t label)
{
	return 0.;
}

float64_t HingeLoss::second_derivative(float64_t z)
{
	return 0;
}

float64_t HingeLoss::get_update(float64_t prediction, float64_t label, float64_t eta_t, float64_t norm)
{
	if (label * prediction >= label * label)
		return 0;
	float64_t err = (label*label - label*prediction)/(label * label);
	float64_t normal = eta_t;
	return label * (normal < err ? normal : err)/norm;
}

float64_t HingeLoss::get_square_grad(float64_t prediction, float64_t label)
{
	return first_derivative(prediction, label);
}
