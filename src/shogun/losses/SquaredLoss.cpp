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
*/

#include <shogun/mathematics/Math.h>
#include <shogun/losses/SquaredLoss.h>

using namespace shogun;

float64_t CSquaredLoss::loss(float64_t prediction, float64_t label)
{
	float64_t example_loss = (prediction - label) * (prediction - label);

	return example_loss;
}

float64_t CSquaredLoss::first_derivative(float64_t prediction, float64_t label)
{
	return 2. * (prediction - label);
}

float64_t CSquaredLoss::second_derivative(float64_t prediction, float64_t label)
{
	return 2;
}
