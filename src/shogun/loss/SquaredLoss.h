/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Shashwat Lal Das, Fernando Iglesias, Yuyu Zhang, Bjoern Esser
 */

#ifndef _SQUAREDLOSS_H__
#define _SQUAREDLOSS_H__

#include <shogun/lib/config.h>

#include <shogun/loss/LossFunction.h>

namespace shogun
{
/** @brief CSquaredLoss implements the
 * squared loss function.
 */
class SquaredLoss: public LossFunction
{
public:
	/**
	 * Constructor
	 */
	SquaredLoss(): LossFunction() {};

	/**
	 * Destructor
	 */
	~SquaredLoss() {};

	/**
	 * Get loss for an example
	 *
	 * @param prediction prediction
	 * @param label label
	 *
	 * @return loss
	 */
	float64_t loss(float64_t prediction, float64_t label);

	/**
	 * Get loss for an example
	 *
	 * @param z where to evaluate the loss
	 *
	 * @return loss
	 */
	float64_t loss(float64_t z);

	/**
	 * Get square of the gradient of the loss function
	 *
	 * @param prediction prediction
	 * @param label label
	 *
	 * @return square of gradient
	 */
	float64_t first_derivative(float64_t prediction, float64_t label);

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
	 * @param prediction prediction
	 * @param label label
	 *
	 * @return second derivative
	 */
	float64_t second_derivative(float64_t prediction, float64_t label);

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
	 * @return L_SQUAREDLOSS
	 */
	virtual ELossType get_loss_type() { return L_SQUAREDLOSS; }

	virtual const char* get_name() const { return "SquaredLoss"; }
};

}

#endif
