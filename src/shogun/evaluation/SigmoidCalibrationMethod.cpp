/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2012 - 2013 Heiko Strathmann
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
 * FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are
 * those
 * of the authors and should not be interpreted as representing official
 * policies,
 * either expressed or implied, of the Shogun Development Team.
 */

#include <shogun/evaluation/SigmoidCalibrationMethod.h>
#include <shogun/machine/Machine.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/Statistics.h>

using namespace shogun;

CSigmoidCalibrationMethod::CSigmoidCalibrationMethod() : CCalibrationMethod()
{
	init();
}

CSigmoidCalibrationMethod::CSigmoidCalibrationMethod(
    SGVector<float64_t> target_values)
    : CCalibrationMethod(target_values)
{
	init();
};

CSigmoidCalibrationMethod::~CSigmoidCalibrationMethod()
{
}

void CSigmoidCalibrationMethod::init()
{
	m_a = 0;
	m_b = 0;
	SG_ADD(&m_a, "m_a", "sigmoid parameter a", MS_NOT_AVAILABLE);
	SG_ADD(&m_b, "m_b", "sigmoid parameter b", MS_NOT_AVAILABLE);
}

SGVector<float64_t>
CSigmoidCalibrationMethod::apply_binary(SGVector<float64_t> values)
{
	// Code borrowed from CBinaryLabels.cpp
	for (index_t i = 0; i < values.vlen; ++i)
	{
		float64_t fApB = values[i] * m_a + m_b;
		values[i] = fApB >= 0 ? CMath::exp(-fApB) / (1.0 + CMath::exp(-fApB))
		                      : 1.0 / (1 + CMath::exp(fApB));
	}
	return values;
}

bool CSigmoidCalibrationMethod::train(SGVector<float64_t> values)
{
	CStatistics::SigmoidParamters params = CStatistics::fit_sigmoid(values);
	m_a = params.a;
	m_b = params.b;

	return true;
}
