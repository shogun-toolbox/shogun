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

#include <shogun/optimization/GradientDescendUpdater.h>
#include <shogun/lib/config.h>
using namespace shogun;

GradientDescendUpdater::GradientDescendUpdater()
	:DescendUpdaterWithCorrection()
{
	init();
}

GradientDescendUpdater::GradientDescendUpdater(LearningRate *learning_rate)
	:DescendUpdaterWithCorrection()
{
	init();
	set_learning_rate(learning_rate);
}

void GradientDescendUpdater::set_learning_rate(LearningRate *learning_rate)
{
	REQUIRE(learning_rate,"Learning_rate must set\n");
	m_learning_rate=learning_rate;
}

GradientDescendUpdater::~GradientDescendUpdater()
{
}

void GradientDescendUpdater::init()
{
	m_learning_rate=NULL;
}

void GradientDescendUpdater::update_context(CMinimizerContext* context)
{
	DescendUpdaterWithCorrection::update_context(context);
	REQUIRE(m_learning_rate,"Learning_rate must set\n");
	REQUIRE(context, "Context must set\n");
	m_learning_rate->update_context(context);
}

void GradientDescendUpdater::load_from_context(CMinimizerContext* context)
{
	DescendUpdaterWithCorrection::load_from_context(context);
	REQUIRE(m_learning_rate,"learning_rate must set\n");
	REQUIRE(context, "context must set\n");
	m_learning_rate->load_from_context(context);
}

float64_t GradientDescendUpdater::get_negative_descend_direction(float64_t variable,
	float64_t gradient, index_t idx)
{
	return m_learning_rate->get_learning_rate(false)*gradient;
}

void GradientDescendUpdater::update_variable(SGVector<float64_t> variable_reference,
	SGVector<float64_t> raw_negative_descend_direction)
{
	REQUIRE(m_learning_rate,"learning_rate must set\n");
	// must call LearningRate::get_learning_rate() here
	// if we want to decay the learning rate at each iteration
	m_learning_rate->get_learning_rate(true);
	DescendUpdaterWithCorrection::update_variable(variable_reference, raw_negative_descend_direction);
}
