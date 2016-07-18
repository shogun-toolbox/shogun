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
#include <shogun/optimization/ElasticNetPenalty.h>

using namespace shogun;

void ElasticNetPenalty::set_l1_ratio(float64_t ratio)
{
	REQUIRE(ratio>0.0 && ratio<1.0, "ratio (%f) must be in (0.0,1.0)", ratio);
	m_l1_ratio=ratio;
}

float64_t ElasticNetPenalty::get_penalty(float64_t variable)
{
	check_ratio();
	float64_t penalty=m_l1_ratio*m_l1_penalty->get_penalty(variable);
	penalty+=(1.0-m_l1_ratio)*m_l2_penalty->get_penalty(variable);
	return penalty;
}

float64_t ElasticNetPenalty::get_penalty_gradient(float64_t variable,
	float64_t gradient_of_variable)
{
	check_ratio();
	float64_t grad=m_l1_ratio*m_l1_penalty->get_penalty_gradient(variable, gradient_of_variable);
	grad+=(1.0-m_l1_ratio)*m_l2_penalty->get_penalty_gradient(variable, gradient_of_variable);
	return grad;
}

void ElasticNetPenalty::update_variable_for_proximity(SGVector<float64_t> variable,
	float64_t proximal_weight)
{
	check_ratio();
	m_l1_penalty->update_variable_for_proximity(variable, proximal_weight*m_l1_ratio);
}

float64_t ElasticNetPenalty::get_sparse_variable(float64_t variable, float64_t penalty_weight)
{
	check_ratio();
	return m_l1_penalty->get_sparse_variable(variable, penalty_weight*m_l1_ratio);
}

void ElasticNetPenalty::check_ratio()
{
	REQUIRE(m_l1_ratio>0, "l1_ratio must set\n");
}

ElasticNetPenalty::~ElasticNetPenalty()
{
	SG_UNREF(m_l1_penalty);
	SG_UNREF(m_l2_penalty);
}

void ElasticNetPenalty::init()
{
	m_l1_ratio=0;
	m_l1_penalty=new L1Penalty();
	m_l2_penalty=new L2Penalty();
	SG_ADD(&m_l1_ratio, "ElasticNetPenalty__m_l1_ratio",
		"l1_ratio in ElasticNetPenalty", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject **) &m_l1_penalty, "ElasticNetPenalty__m_l1_penalty",
		"l1_penalty in ElasticNetPenalty", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject **) &m_l2_penalty, "ElasticNetPenalty__m_l2_penalty",
		"l2_penalty in ElasticNetPenalty", MS_NOT_AVAILABLE);
}
