/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Fernando Iglesias, Shashwat Lal Das
 */

#include <shogun/mathematics/Math.h>
#include <shogun/loss/SquaredLoss.h>

using namespace shogun;

float64_t SquaredLoss::loss(float64_t prediction, float64_t label)
{
	return (prediction - label) * (prediction - label);
}

float64_t SquaredLoss::loss(float64_t z)
{
	return z*z;
}

float64_t SquaredLoss::first_derivative(float64_t prediction, float64_t label)
{
	return 2. * (prediction - label);
}

float64_t SquaredLoss::first_derivative(float64_t z)
{
	return 2. * z;
}

float64_t SquaredLoss::second_derivative(float64_t prediction, float64_t label)
{
	return 2;
}

float64_t SquaredLoss::second_derivative(float64_t z)
{
	return 2;
}

float64_t SquaredLoss::get_update(float64_t prediction, float64_t label, float64_t eta_t, float64_t norm)
{
    if (eta_t < 1e-6)
    {
      /* When exp(-eta_t)~= 1 we replace 1-exp(-eta_t)
       * with its first order Taylor expansion around 0
       * to avoid catastrophic cancellation.
       */
      return (label - prediction)*eta_t/norm;
    }
    return (label - prediction)*(1-exp(-eta_t))/norm;
}

float64_t SquaredLoss::get_square_grad(float64_t prediction, float64_t label)
{
	return (prediction - label) * (prediction - label);
}
