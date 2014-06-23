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

#include <shogun/loss/HuberLoss.h>
#include <shogun/base/Parameter.h>

using namespace shogun;

CHuberLoss::CHuberLoss(float64_t delta)
: CLossFunction()
{
	init();
	m_delta=delta;
}

float64_t CHuberLoss::loss(float64_t prediction, float64_t label)
{
	return loss(prediction-label);
}

float64_t CHuberLoss::loss(float64_t z)
{
	if (CMath::abs(z)<m_delta)
		return z*z;
	else
		return m_delta*(CMath::abs(z)-m_delta/2.0);
}

float64_t CHuberLoss::first_derivative(float64_t prediction, float64_t label)
{
	return first_derivative(prediction-label);
}

float64_t CHuberLoss::first_derivative(float64_t z)
{
	if (CMath::abs(z)<m_delta)
		return 2*z;
	else
		return (z>0)?m_delta:-m_delta;
}

float64_t CHuberLoss::second_derivative(float64_t prediction, float64_t label)
{
	return second_derivative(prediction-label);
}

float64_t CHuberLoss::second_derivative(float64_t z)
{
	if (CMath::abs(z)<m_delta)
		return 2;
	else
		return 0;
}

float64_t CHuberLoss::get_update(float64_t prediction, float64_t label, float64_t eta_t, float64_t norm)
{
	SG_NOTIMPLEMENTED;
	return 0;
}

float64_t CHuberLoss::get_square_grad(float64_t prediction, float64_t label)
{
	SG_NOTIMPLEMENTED;
	return 0;
}

void CHuberLoss::init()
{
	m_delta=0;

	SG_ADD(&m_delta,"m_delta","delta",MS_NOT_AVAILABLE);
}
