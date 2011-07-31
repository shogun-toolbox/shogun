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

#include <shogun/loss/HingeLoss.h>

using namespace shogun;

float64_t CHingeLoss::loss(float64_t prediction, float64_t label)
{
	float64_t e = 1 - label * prediction;

	return (e > 0) ? e : 0;
}

float64_t CHingeLoss::first_derivative(float64_t prediction, float64_t label)
{
	return (label * prediction >= label * label) ? 0 : -label;
}

float64_t CHingeLoss::second_derivative(float64_t prediction, float64_t label)
{
	return 0.;
}
