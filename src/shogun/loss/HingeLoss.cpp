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

#include <loss/HingeLoss.h>
#include <mathematics/Math.h>

using namespace shogun;

float64_t CHingeLoss::loss(float64_t prediction, float64_t label)
{
	float64_t e = 1 - label * prediction;

	return (e > 0) ? e : 0;
}

float64_t CHingeLoss::loss(float64_t z)
{
	return CMath::max(0.0, z);
}

float64_t CHingeLoss::first_derivative(float64_t prediction, float64_t label)
{
	return (label * prediction >= label * label) ? 0 : -label;
}

float64_t CHingeLoss::first_derivative(float64_t z)
{
	return z > 0.0 ? 1.0 : 0.0;
}

float64_t CHingeLoss::second_derivative(float64_t prediction, float64_t label)
{
	return 0.;
}

float64_t CHingeLoss::second_derivative(float64_t z)
{
	return 0;
}

float64_t CHingeLoss::get_update(float64_t prediction, float64_t label, float64_t eta_t, float64_t norm)
{
	if (label * prediction >= label * label)
		return 0;
	float64_t err = (label*label - label*prediction)/(label * label);
	float64_t normal = eta_t;
	return label * (normal < err ? normal : err)/norm;
}

float64_t CHingeLoss::get_square_grad(float64_t prediction, float64_t label)
{
	return first_derivative(prediction, label);
}
