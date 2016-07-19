/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2015 Wu Lin
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
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
 *
 */

#include <shogun/optimization/InverseScalingLearningRate.h>
#include <shogun/mathematics/Math.h>
#include <shogun/base/Parameter.h>

using namespace shogun;
float64_t InverseScalingLearningRate::get_learning_rate(int32_t iter_counter)
{
	REQUIRE(iter_counter,"Iter_counter (%d) must be positive\n", iter_counter);
	return m_initial_learning_rate/CMath::pow(m_intercept+m_slope*iter_counter,m_exponent);
}

void InverseScalingLearningRate::set_initial_learning_rate(float64_t initial_learning_rate)
{
	REQUIRE(initial_learning_rate>0.0, "Initial learning rate (%f) should be positive\n",
		initial_learning_rate);
	m_initial_learning_rate=initial_learning_rate;
}

void InverseScalingLearningRate::set_exponent(float64_t exponent)
{
	REQUIRE(exponent>0.0, "Exponent (%f) should be positive\n", exponent);
	m_exponent=exponent;
}

void InverseScalingLearningRate::set_slope(float64_t slope)
{
	REQUIRE(slope>0.0,"Slope (%f) should be positive\n", slope);
	m_slope=slope;
}

void InverseScalingLearningRate::set_intercept(float64_t intercept)
{
	REQUIRE(intercept>=0, "Intercept (%f) should be non-negative\n",
		intercept);
	m_intercept=intercept;
}
void InverseScalingLearningRate::init()
{
	m_exponent=0.5;
	m_initial_learning_rate=1.0;
	m_intercept=0.0;
	m_slope=1.0;
	SG_ADD(&m_slope, "InverseScalingLearningRate__m_slope",
		"slope in InverseScalingLearningRate", MS_NOT_AVAILABLE);
	SG_ADD(&m_exponent, "InverseScalingLearningRate__m_exponent",
		"exponent in InverseScalingLearningRate", MS_NOT_AVAILABLE);
	SG_ADD(&m_intercept, "InverseScalingLearningRate__m_intercept",
		"intercept in InverseScalingLearningRate", MS_NOT_AVAILABLE);
	SG_ADD(&m_initial_learning_rate, "InverseScalingLearningRate__m_initial_learning_rate",
		"initial_learning_rate in InverseScalingLearningRate", MS_NOT_AVAILABLE);
}
