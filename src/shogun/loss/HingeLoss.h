/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Shashwat Lal Das, Fernando Iglesias, Yuyu Zhang, Bjoern Esser
 */

#ifndef _HINGELOSS_H__
#define _HINGELOSS_H__

#include <shogun/lib/config.h>

#include <shogun/loss/LossFunction.h>

namespace shogun
{
/** @brief HingeLoss implements the hinge
 * loss function.
 */
class HingeLoss: public LossFunction
{
public:
	/**
	 * Constructor
	 */
	HingeLoss(): LossFunction() {};

	/**
	 * Destructor
	 */
	~HingeLoss() override {};

	/**
	 * Get loss for an example
	 *
	 * @param prediction prediction
	 * @param label label
	 *
	 * @return loss
	 */
	float64_t loss(float64_t prediction, float64_t label) override;

	/**
	 * Get loss for an example. The definition used for the
	 * hinge loss computed by this method is f(x) = max(0, x).
	 *
	 * @param z where to evaluate the loss
	 *
	 * @return loss
	 */
	float64_t loss(float64_t z) override;

	/**
	 * Get first derivative of the loss function
	 *
	 * @param prediction prediction
	 * @param label label
	 *
	 * @return first derivative
	 */
	float64_t first_derivative(float64_t prediction, float64_t label) override;

	/**
	 * Get first derivative of the loss function
	 *
	 * @param z where to evaluate the derivative of the loss
	 *
	 * @return first derivative
	 */
	float64_t first_derivative(float64_t z) override;

	/**
	 * Get second derivative of the loss function
	 *
	 * @param prediction prediction
	 * @param label label
	 *
	 * @return second derivative
	 */
	float64_t second_derivative(float64_t prediction, float64_t label) override;

	/**
	 * Get second derivative of the loss function
	 *
	 * @param z where to evaluate the second derivative of the loss
	 *
	 * @return second derivative
	 */
	float64_t second_derivative(float64_t z) override;

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
	float64_t get_update(float64_t prediction, float64_t label, float64_t eta_t, float64_t norm) override;

	/**
	 * Get square of gradient, used for adaptive learning
	 *
	 * @param prediction prediction
	 * @param label label
	 *
	 * @return square of gradient
	 */
	float64_t get_square_grad(float64_t prediction, float64_t label) override;

	/**
	 * Return type of loss
	 *
	 * @return L_HINGELOSS
	 */
	ELossType get_loss_type() override { return L_HINGELOSS; }

	const char* get_name() const override { return "HingeLoss"; }
};

}

#endif
