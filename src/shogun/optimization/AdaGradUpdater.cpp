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

#include <shogun/optimization/AdaGradUpdater.h>
#include <shogun/mathematics/Math.h>
#include <shogun/base/Parameter.h>
using namespace shogun;

AdaGradUpdater::AdaGradUpdater()
	:DescendUpdaterWithCorrection()
{
	init();
}

AdaGradUpdater::AdaGradUpdater(float64_t learning_rate,float64_t epsilon)
	:DescendUpdaterWithCorrection()
{
	init();
	set_learning_rate(learning_rate);
	set_epsilon(epsilon);
}

void AdaGradUpdater::set_learning_rate(float64_t learning_rate)
{
	REQUIRE(learning_rate>0,"Learning_rate (%f) must be positive\n",
		learning_rate);
	m_build_in_learning_rate=learning_rate;
}

void AdaGradUpdater::set_epsilon(float64_t epsilon)
{
	REQUIRE(epsilon>=0,"Epsilon (%f) must be non-negative\n",
		epsilon);
	m_epsilon=epsilon;
}

AdaGradUpdater::~AdaGradUpdater() { }

void AdaGradUpdater::init()
{
	m_epsilon=1e-6;
	m_build_in_learning_rate=1.0;
	m_gradient_accuracy=SGVector<float64_t>();

	SG_ADD(&m_epsilon, "AdaGradUpdater__m_epsilon",
		"epsilon in AdaGradUpdater", MS_NOT_AVAILABLE);
	SG_ADD(&m_build_in_learning_rate, "AdaGradUpdater__m_build_in_learning_rate",
		"m_build_in_learning_rate in AdaGradUpdater", MS_NOT_AVAILABLE);
	SG_ADD(&m_gradient_accuracy, "AdaGradUpdater__m_gradient_accuracy",
		"gradient_accuracy in AdaGradUpdater", MS_NOT_AVAILABLE);
}

float64_t AdaGradUpdater::get_negative_descend_direction(float64_t variable,
	float64_t gradient, index_t idx, float64_t learning_rate)
{
	REQUIRE(idx>=0 && idx<m_gradient_accuracy.vlen, "The index (%d) is invalid\n", idx);
	float64_t scale=m_gradient_accuracy[idx]+gradient*gradient;
	m_gradient_accuracy[idx]=scale;
	float64_t res=m_build_in_learning_rate*gradient/CMath::sqrt(scale+m_epsilon);
	return res;
}

void AdaGradUpdater::update_variable(SGVector<float64_t> variable_reference,
	SGVector<float64_t> raw_negative_descend_direction, float64_t learning_rate)
{
	REQUIRE(variable_reference.vlen==raw_negative_descend_direction.vlen,
		"The length of variable (%d) and the length of negative descend direction (%d) do not match\n",
		variable_reference.vlen, raw_negative_descend_direction.vlen);
	if(m_gradient_accuracy.vlen==0)
	{
		m_gradient_accuracy=SGVector<float64_t>(variable_reference.vlen);
		m_gradient_accuracy.set_const(0.0);
	}
	DescendUpdaterWithCorrection::update_variable(variable_reference,
		raw_negative_descend_direction, learning_rate);
}
