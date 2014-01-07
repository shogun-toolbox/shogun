/*
  Copyright (c) 2009 Yahoo! Inc.  All rights reserved.  The copyrights
  embodied in the content of this file are licensed under the BSD
  (revised) open source license.

  Copyright (c) 2011 Berlin Institute of Technology and Max-Planck-Society.

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 3 of the License, or
  (at your option) any later version.

  Modifications (w) 2011 Shashwat Lal Das
  Modifications (w) 2012 Fernando José Iglesias García
*/

#include <mathematics/Math.h>
#include <loss/SquaredLoss.h>

using namespace shogun;

float64_t CSquaredLoss::loss(float64_t prediction, float64_t label)
{
	return (prediction - label) * (prediction - label);
}

float64_t CSquaredLoss::loss(float64_t z)
{
	return z*z;
}

float64_t CSquaredLoss::first_derivative(float64_t prediction, float64_t label)
{
	return 2. * (prediction - label);
}

float64_t CSquaredLoss::first_derivative(float64_t z)
{
	return 2. * z;
}

float64_t CSquaredLoss::second_derivative(float64_t prediction, float64_t label)
{
	return 2;
}

float64_t CSquaredLoss::second_derivative(float64_t z)
{
	return 2;
}

float64_t CSquaredLoss::get_update(float64_t prediction, float64_t label, float64_t eta_t, float64_t norm)
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

float64_t CSquaredLoss::get_square_grad(float64_t prediction, float64_t label)
{
	return (prediction - label) * (prediction - label);
}
