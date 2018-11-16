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

#include <shogun/optimization/L1Penalty.h>
#include <shogun/mathematics/Math.h>
using namespace shogun;

float64_t L1Penalty::get_penalty(float64_t variable)
{
	return CMath::abs(variable);
}

void L1Penalty::set_rounding_epsilon(float64_t epsilon)
{
	REQUIRE(epsilon>=0,"Rounding epsilon (%f) should be non-negative\n", epsilon);
	m_rounding_epsilon=epsilon;
}

void L1Penalty::update_variable_for_proximity(SGVector<float64_t> variable,
	float64_t proximal_weight)
{
	for(index_t idx=0; idx<variable.vlen; idx++)
		variable[idx]=get_sparse_variable(variable[idx], proximal_weight);
}

float64_t L1Penalty::get_sparse_variable(float64_t variable, float64_t penalty_weight)
{
	if (variable>0.0)
	{
		variable-=penalty_weight;
		if (variable<0.0)
			variable=0.0;
	}
	else
	{
		variable+=penalty_weight;
		if (variable>0.0)
			variable=0.0;
	}
	if (CMath::abs(variable)<m_rounding_epsilon)
		variable=0.0;
	return variable;
}

void L1Penalty::init()
{
	m_rounding_epsilon=1e-8;
	SG_ADD(&m_rounding_epsilon, "L1Penalty__m_rounding_epsilon",
		"rounding_epsilon in L1Penalty");
}
