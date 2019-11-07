/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Shashwat Lal Das, Fernando Iglesias, Yuyu Zhang, Saurabh Goyal, 
 *          Bjoern Esser
 */

#ifndef _LOSSFUNCTION_H__
#define _LOSSFUNCTION_H__

#include <shogun/lib/config.h>

#include <shogun/base/SGObject.h>
#include <shogun/lib/common.h>

namespace shogun
{
	/// shogun loss type
	enum ELossType
	{
		L_HINGELOSS = 0,
		L_SMOOTHHINGELOSS = 10,
		L_SQUAREDHINGELOSS = 20,
		L_SQUAREDLOSS = 30,
		L_EXPONENTIALLOSS = 40,
		L_ABSOLUTEDEVIATIONLOSS = 50,
		L_HUBERLOSS = 60,
		L_LOGLOSS = 100,
		L_LOGLOSSMARGIN = 110
	};
}

namespace shogun
{
/** @brief Class CLossFunction is the base class of
 * all loss functions.
 *
 * The class provides the loss for one example,
 * first and second derivates of the loss function,
 * (used very commonly) the square of the gradient and
 * the importance-aware weight update for the function.
 * (used mainly for VW)
 *
 * Refer: Online Importance Weight Aware Updates,
 * Nikos Karampatziakis, John Langford
 * http://arxiv.org/abs/1011.1576
 */
class LossFunction: public SGObject
{
public:

	/**
	 * Constructor
	 */
	LossFunction(): SGObject() {}

	/**
	 * Destructor
	 */
	virtual ~LossFunction() {};

	/**
	 * Get loss for an example
	 *
	 * @param prediction prediction
	 * @param label label
	 *
	 * @return loss
	 */
	virtual float64_t loss(float64_t prediction, float64_t label)
	{
		return loss(prediction * label);
	}

	/**
	 * Get loss for an example
	 *
	 * @param z where to evaluate the loss
	 *
	 * @return loss
	 */
	virtual float64_t loss(float64_t z) = 0;

	/**
	 * Get first derivative of the loss function
	 *
	 * @param prediction prediction
	 * @param label label
	 *
	 * @return first derivative
	 */
	virtual float64_t first_derivative(float64_t prediction, float64_t label)
	{
		return loss(prediction * label);
	}

	/**
	 * Get first derivative of the loss function
	 *
	 * @param z where to evaluate the derivative of the loss
	 *
	 * @return first derivative
	 */
	virtual float64_t first_derivative(float64_t z) = 0;

	/**
	 * Get second derivative of the loss function
	 *
	 * @param prediction prediction
	 * @param label label
	 *
	 * @return second derivative
	 */
	virtual float64_t second_derivative(float64_t prediction, float64_t label)
	{
		return loss(prediction * label);
	}

	/**
	 * Get second derivative of the loss function
	 *
	 * @param z where to evaluate the second derivative of the loss
	 *
	 * @return second derivative
	 */
	virtual float64_t second_derivative(float64_t z) = 0;

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

	/**
	 * Return the name of the object
	 *
	 * @return LossFunction
	 */
	virtual const char* get_name() const { return "LossFunction"; }
};
}
#endif // _LOSSFUNCTION_H__
