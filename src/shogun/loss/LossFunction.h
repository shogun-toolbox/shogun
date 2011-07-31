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
#include <math.h>

namespace shogun
{
	/// shogun loss type
	enum ELossType
	{
		L_HINGELOSS = 0,
		L_SMOOTHHINGELOSS = 10,
		L_SQUAREDHINGELOSS = 20,
		L_SQUAREDLOSS = 30,
		L_LOGLOSS = 100,
		L_LOGLOSSMARGIN = 110
	};
}

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

	/**
	 * Get importance aware weight update for this loss function
	 *
	 * @param prediction prediction
	 * @param label label
	 * @param eta_t learning rate at update number t
	 * @param norm scale value
	 *
	 * @return update
	 */
	virtual float64_t get_update(float64_t prediction, float64_t label, float64_t eta_t, float64_t norm) = 0;

	/**
	 * Get square of gradient, used for adaptive learning
	 *
	 * @param prediction prediction
	 * @param label label
	 *
	 * @return square of gradient
	 */
	virtual float64_t get_square_grad(float64_t prediction, float64_t label) = 0;

	/**
	 * Get loss type
	 *
	 * abstract base method
	 *
	 * @return loss type as enum
	 */
	virtual ELossType get_loss_type()=0;

	virtual const char* get_name() const { return "LossFunction"; }
};
}
#endif
