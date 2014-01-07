/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (c) 2011 Berlin Institute of TechnosmoothHingey and Max-Planck-Society.
 */

#ifndef _SMOOTHHINGELOSS_H__
#define _SMOOTHHINGELOSS_H__

#include <loss/LossFunction.h>

namespace shogun
{
/** @brief CSmoothHingeLoss implements
 * the smooth hinge loss function.
 */
class CSmoothHingeLoss: public CLossFunction
{
public:
	/**
	 * Constructor
	 */
	CSmoothHingeLoss(): CLossFunction() {};

	/**
	 * Destructor
	 */
	~CSmoothHingeLoss() {};

	/**
	 * Get loss for an example
	 *
	 * @param z where to evaluate the loss
	 *
	 * @return loss
	 */
	float64_t loss(float64_t z);

	/**
	 * Get first derivative of the loss function
	 *
	 * @param z where to evaluate the derivative of the loss
	 *
	 * @return first derivative
	 */
	float64_t first_derivative(float64_t z);

	/**
	 * Get second derivative of the loss function
	 *
	 * @param z where to evaluate the second derivative of the loss
	 *
	 * @return second derivative
	 */
	float64_t second_derivative(float64_t z);

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
	virtual float64_t get_update(float64_t prediction, float64_t label, float64_t eta_t, float64_t norm);

	/**
	 * Get square of gradient, used for adaptive learning
	 *
	 * @param prediction prediction
	 * @param label label
	 *
	 * @return square of gradient
	 */
	virtual float64_t get_square_grad(float64_t prediction, float64_t label);

	/**
	 * Return loss type
	 *
	 * @return L_SMOOTHHINGELOSS
	 */
	virtual ELossType get_loss_type() { return L_SMOOTHHINGELOSS; }

	virtual const char* get_name() const { return "SmoothHingeLoss"; }
};

}

#endif
