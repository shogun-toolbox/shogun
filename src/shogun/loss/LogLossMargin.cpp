/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Fernando Iglesias, Shashwat Lal Das, Thoralf Klein, 
 *          Soeren Sonnenburg
 */

#include <shogun/loss/LogLossMargin.h>
#include <shogun/io/SGIO.h>

using namespace shogun;

float64_t LogLossMargin::loss(float64_t z)
{
	if (z >= 1)
		return log(1+exp(1-z));

	return 1-z + log(1+exp(z-1));
}

float64_t LogLossMargin::first_derivative(float64_t z)
{
	if (z < 1)
		return -1 / (exp(z-1) + 1);

	float64_t ez = exp(1-z);
	return -ez / (ez + 1);
}

float64_t LogLossMargin::second_derivative(float64_t z)
{
	float64_t ez = exp(z-1);

	return ez / (ez + 1)*(ez + 1);
}

float64_t LogLossMargin::get_update(float64_t prediction, float64_t label, float64_t eta_t, float64_t norm)
{
	not_implemented(SOURCE_LOCATION);
	return -1;
}

float64_t LogLossMargin::get_square_grad(float64_t prediction, float64_t label)
{
	not_implemented(SOURCE_LOCATION);
	return -1;
}
