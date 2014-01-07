/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (c) 2011 Berlin Institute of Technology and Max-Planck-Society.
 */

#include <loss/LogLoss.h>

using namespace shogun;

float64_t CLogLoss::loss(float64_t z)
{
	return (z >= 0) ? log(1 + exp(-z)) : -z + log(1 + exp(z));
}

float64_t CLogLoss::first_derivative(float64_t z)
{
	if (z < 0)
		return -1 / (exp(z) + 1);

	float64_t ez = exp(-z);
	return -ez / (ez + 1);
}

float64_t CLogLoss::second_derivative(float64_t z)
{
	float64_t ez = exp(z);
	return ez / (ez*(ez + 2) + 1);
}

float64_t CLogLoss::get_update(float64_t prediction, float64_t label, float64_t eta_t, float64_t norm)
{
	float64_t w,x;
	float64_t d = exp(label * prediction);
	if(eta_t < 1e-6){
		/* As with squared loss, for small eta_t we replace the update
		 * with its first order Taylor expansion to avoid numerical problems
		 */
		return label*eta_t/((1+d)*norm);
	}
	x = eta_t + label*prediction + d;

	/* This piece of code is approximating W(exp(x))-x.
	 * W is the Lambert W function: W(z)*exp(W(z))=z.
	 * The absolute error of this approximation is less than 9e-5.
	 * Faster/better approximations can be substituted here.
	 */
	float64_t W = x>=1. ? 0.86*x+0.01 : exp(0.8*x-0.65); //initial guess
	float64_t r = x>=1. ? x-log(W)-W : 0.2*x+0.65-W; //residual
	float64_t t = 1.+W;
	float64_t u = 2.*t*(t+2.*r/3.); //magic
	w = W*(1.+r/t*(u-r)/(u-2.*r))-x; //more magic

	return -(label*w+prediction)/norm;
}

float64_t CLogLoss::get_square_grad(float64_t prediction, float64_t label)
{
	float64_t d = CLossFunction::first_derivative(prediction, label);
	return d*d;
}

