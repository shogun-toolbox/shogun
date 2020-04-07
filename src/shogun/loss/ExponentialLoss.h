/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Parijat Mazumdar
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 */

#ifndef _EXPONENTIALLOSS_H__
#define _EXPONENTIALLOSS_H__

#include <shogun/lib/config.h>

#include <shogun/loss/LossFunction.h>

namespace shogun
{
/** @brief CExponentialLoss implements the exponential loss function. \n
 * \f$L(y_i,f(x_i)) = \exp^{-y_if(x_i)}\f$
 */
class ExponentialLoss: public LossFunction
{
public:
	/** default constructor */
	ExponentialLoss(): LossFunction() {};

	/** Destructor */
	~ExponentialLoss() override {};

	/** Get loss for an example
	 *
	 * @param prediction predicted label \f$f(x_i)\f$
	 * @param label actual label \f$y_i\f$
	 * @return loss
	 */
	float64_t loss(float64_t prediction, float64_t label) override;

	/** Get loss for an example
	 *
	 * @param z where to evaluate the loss
	 * @return loss
	 */
	float64_t loss(float64_t z) override;

	/** Get first derivative of the loss function
	 *
	 * @param prediction predicted label \f$f(x_i)\f$
	 * @param label actual label \f$y_i\f$
	 * @return gradient
	 */
	float64_t first_derivative(float64_t prediction, float64_t label) override;

	/** Get first derivative of the loss function
	 *
	 * @param z where to evaluate the derivative of the loss
	 * @return first derivative
	 */
	float64_t first_derivative(float64_t z) override;

	/** Get second derivative of the loss function
	 *
	 * @param prediction prediction
	 * @param label label
	 * @return second derivative
	 */
	float64_t second_derivative(float64_t prediction, float64_t label) override;

	/** Get second derivative of the loss function
	 *
	 * @param z where to evaluate the second derivative of the loss
	 * @return second derivative
	 */
	float64_t second_derivative(float64_t z) override;

	/** Get importance aware weight update for this loss function
	 * NOT IMPLEMENTED
	 *
	 * @param prediction prediction
	 * @param label label
	 * @param eta_t learning rate at update number t
	 * @param norm scale value
	 * @return update
	 */
	float64_t get_update(float64_t prediction, float64_t label, float64_t eta_t, float64_t norm) override;

	/** Get square of gradient, used for adaptive learning
	 * NOT IMPLEMENTED
	 *
	 * @param prediction prediction
	 * @param label label
	 * @return square of gradient
	 */
	float64_t get_square_grad(float64_t prediction, float64_t label) override;

	/** Return loss type
	 *
	 * @return L_EXPONENTIALLOSS
	 */
	ELossType get_loss_type() override { return L_EXPONENTIALLOSS; }

	/** Return name
	 *
	 * @return ExponentialLoss
	 */
	const char* get_name() const override { return "ExponentialLoss"; }
};

} /* shogun */

#endif /* _EXPONENTIAL_LOSS__ */
