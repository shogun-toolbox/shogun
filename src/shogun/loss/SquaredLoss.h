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

#ifndef _SQUAREDLOSS_H__
#define _SQUAREDLOSS_H__

#include <shogun/loss/LossFunction.h>

namespace shogun
{

class CSquaredLoss: public CLossFunction
{
public:
	/**
	 * Constructor
	 */
	CSquaredLoss() {};

	/**
	 * Destructor
	 */
	~CSquaredLoss() {};

	/**
	 * Get loss for an example
	 *
	 * @param prediction prediction
	 * @param label label
	 *
	 * @return loss
	 */
	virtual float64_t loss(float64_t prediction, float64_t label);

	/**
	 * Get square of the gradient of the loss function
	 *
	 * @param prediction prediction
	 * @param label label
	 *
	 * @return square of gradient
	 */
	virtual float64_t first_derivative(float64_t prediction, float64_t label);

	/**
	 * Get second derivative of the loss function
	 *
	 * @param prediction prediction
	 * @param label label
	 *
	 * @return second derivative
	 */
	virtual float64_t second_derivative(float64_t prediction, float64_t label);

	virtual const char* get_name() const { return "SquaredLoss"; }
};

}

#endif
