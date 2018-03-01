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

#include <shogun/optimization/AdamUpdater.h>
#include <shogun/mathematics/Math.h>
#include <shogun/base/Parameter.h>

using namespace shogun;

AdamUpdater::AdamUpdater()
	:DescendUpdaterWithCorrection()
{
	init();
}

AdamUpdater::AdamUpdater(float64_t learning_rate,float64_t epsilon,
	float64_t first_moment_decay_factor,
	float64_t second_moment_decay_factor)
	:DescendUpdaterWithCorrection()
{
	init();
	set_learning_rate(learning_rate);
	set_epsilon(epsilon);
	set_first_moment_decay_factor(first_moment_decay_factor);
	set_second_moment_decay_factor(second_moment_decay_factor);
}

void AdamUpdater::set_learning_rate(float64_t learning_rate)
{
	REQUIRE(learning_rate>0,"Learning_rate (%f) must be positive\n",
		learning_rate);
	m_log_learning_rate = std::log(learning_rate);
}

void AdamUpdater::set_epsilon(float64_t epsilon)
{
	REQUIRE(epsilon>0,"Epsilon (%f) must be non-negative\n",
		epsilon);
	m_epsilon=epsilon;
}

void AdamUpdater::set_first_moment_decay_factor(float64_t decay_factor)
{
	REQUIRE(decay_factor>0.0 && decay_factor<=1.0,
		"Decay factor (%f) for first moment must in (0,1]\n",
		decay_factor);
	m_decay_factor_first_moment=decay_factor;
}

void AdamUpdater::set_second_moment_decay_factor(float64_t decay_factor)
{
	REQUIRE(decay_factor>0.0 && decay_factor<=1.0,
		"Decay factor (%f) for second moment must in (0,1]\n",
		decay_factor);
	m_decay_factor_second_moment=decay_factor;
}

AdamUpdater::~AdamUpdater() { }

void AdamUpdater::init()
{
	m_decay_factor_first_moment=0.9;
	m_decay_factor_second_moment=0.999;
	m_epsilon=1e-8;
	m_log_learning_rate = std::log(0.001);
	m_iteration_counter=0;
	m_log_scale_pre_iteration=0;
	m_gradient_first_moment=SGVector<float64_t>();
	m_gradient_second_moment=SGVector<float64_t>();

	SG_ADD(&m_decay_factor_first_moment, "AdamUpdater__m_decay_factor_first_moment",
		"decay_factor_first_moment in AdamUpdater", MS_NOT_AVAILABLE);
	SG_ADD(&m_decay_factor_second_moment, "AdamUpdater__m_decay_factor_second_moment",
		"decay_factor_second_moment in AdamUpdater", MS_NOT_AVAILABLE);
	SG_ADD(&m_gradient_first_moment, "AdamUpdater__m_gradient_first_moment",
		"m_gradient_first_moment in AdamUpdater", MS_NOT_AVAILABLE);
	SG_ADD(&m_gradient_second_moment, "AdamUpdater__m_gradient_second_moment",
		"m_gradient_second_moment in AdamUpdater", MS_NOT_AVAILABLE);
	SG_ADD(&m_epsilon, "AdamUpdater__m_epsilon",
		"epsilon in AdamUpdater", MS_NOT_AVAILABLE);
	SG_ADD(&m_log_scale_pre_iteration, "AdamUpdater__m_log_scale_pre_iteration",
		"log_scale_pre_iteration in AdamUpdater", MS_NOT_AVAILABLE);
	SG_ADD(&m_log_learning_rate, "AdamUpdater__m_log_learning_rate",
		"m_log_learning_rate in AdamUpdater", MS_NOT_AVAILABLE);
	SG_ADD(&m_iteration_counter, "AdamUpdater__m_iteration_counter",
		"m_iteration_counter in AdamUpdater", MS_NOT_AVAILABLE);
}

float64_t AdamUpdater::get_negative_descend_direction(float64_t variable,
	float64_t gradient, index_t idx, float64_t learning_rate)
{
	REQUIRE(idx>=0 && idx<m_gradient_first_moment.vlen, "");
	REQUIRE(idx>=0 && idx<m_gradient_second_moment.vlen, "");

	float64_t scale_first_moment=m_decay_factor_first_moment*m_gradient_first_moment[idx]+
		(1.0-m_decay_factor_first_moment)*gradient;
	m_gradient_first_moment[idx]=scale_first_moment;


	float64_t scale_second_moment=m_decay_factor_second_moment*m_gradient_second_moment[idx]+
		(1.0-m_decay_factor_second_moment)*gradient*gradient;
	m_gradient_second_moment[idx]=scale_second_moment;

	float64_t res=CMath::exp(m_log_scale_pre_iteration)*scale_first_moment/(CMath::sqrt(scale_second_moment)+m_epsilon);
	return res;
}

void AdamUpdater::update_variable(SGVector<float64_t> variable_reference,
	SGVector<float64_t> raw_negative_descend_direction, float64_t learning_rate)
{
	REQUIRE(variable_reference.vlen==raw_negative_descend_direction.vlen, "");
	if(m_gradient_first_moment.vlen==0)
	{
		m_gradient_first_moment=SGVector<float64_t>(variable_reference.vlen);
		m_gradient_first_moment.set_const(0.0);

		m_gradient_second_moment=SGVector<float64_t>(m_gradient_first_moment.vlen);
		m_gradient_second_moment.set_const(0.0);
	}

	m_iteration_counter++;
	m_log_scale_pre_iteration =
	    m_log_learning_rate +
	    0.5 * std::log(
	              1.0 - CMath::pow(
	                        m_decay_factor_second_moment,
	                        (float64_t)m_iteration_counter)) -
	    std::log(
	        1.0 -
	        CMath::pow(
	            m_decay_factor_first_moment, (float64_t)m_iteration_counter));

	DescendUpdaterWithCorrection::update_variable(variable_reference, raw_negative_descend_direction,
		learning_rate);
}
