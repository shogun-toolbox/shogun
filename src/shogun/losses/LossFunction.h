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

#ifndef _LOSSFUNCTION_H__
#define _LOSSFUNCTION_H__

#include <shogun/base/SGObject.h>
#include <shogun/lib/common.h>

namespace shogun
{

class CLossFunction: public CSGObject
{
public:
	/**
	 * Destructor
	 */
	virtual ~CLossFunction() {};

	/**
	 * Get loss for an example
	 *
	 * @param prediction prediction
	 * @param label label
	 *
	 * @return loss
	 */
	virtual float64_t loss(float64_t prediction, float64_t label) = 0;

	/**
	 * Get the updated learning rate for an example
	 *
	 * @param prediction prediction
	 * @param label label
	 * @param eta_t learning rate
	 * @param norm norm
	 *
	 * @return update
	 */
	virtual float64_t get_update(float64_t prediction, float64_t label,
				     float64_t eta_t, float64_t norm) = 0;

	/**
	 * Get square of the gradient of the loss function
	 *
	 * @param prediction prediction
	 * @param label label
	 *
	 * @return square of gradient
	 */
	virtual float64_t square_grad(float64_t prediction, float64_t label) = 0;

	/**
	 * Get first derivative of the loss function
	 *
	 * @param prediction prediction
	 * @param label label
	 *
	 * @return first derivative
	 */
	virtual float64_t first_derivative(float64_t prediction, float64_t label) = 0;

	/**
	 * Get second derivative of the loss function
	 *
	 * @param prediction prediction
	 * @param label label
	 *
	 * @return second derivative
	 */
	virtual float64_t second_derivative(float64_t prediction, float64_t label) = 0;

	virtual const char* get_name() const { return "LossFunction"; }
};
}
#endif
