/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Shashwat Lal Das, Fernando Iglesias, Yuyu Zhang, Bjoern Esser, 
 *          Sergey Lisitsyn
 */

#ifndef _SQUAREDHINGELOSS_H__
#define _SQUAREDHINGELOSS_H__

#include <shogun/lib/config.h>

#include <shogun/loss/LossFunction.h>

namespace shogun
{
/** @brief Class CSquaredHingeLoss implements a
 * squared hinge loss function.
 */
class SquaredHingeLoss: public LossFunction
{
public:
	/**
	 * Constructor
	 */
	SquaredHingeLoss(): LossFunction() {};

	/**
	 * Destructor
	 */
	~SquaredHingeLoss() override {};

	/**
	 * Get loss for an example
	 *
	 * @param z where to evaluate the loss
	 *
	 * @return loss
	 */
	float64_t loss(float64_t z) override;

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
	 * Return loss type
	 *
	 * @return L_SQUAREDHINGELOSS
	 */
	ELossType get_loss_type() override { return L_SQUAREDHINGELOSS; }

	/**
	 * Return the name of the object
	 *
	 * @return SquaredHingeLoss
	 */
	const char* get_name() const override { return "SquaredHingeLoss"; }
};

}

#endif // _SQUAREDHINGELOSS_H__
