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

#include <shogun/optimization/L1PenaltyForTG.h>
using namespace shogun;

void L1PenaltyForTG::update_variable_for_proximity(SGVector<float64_t> variable,
	float64_t proximal_weight)
{
	if(m_q.vlen==0)
	{
		m_q=SGVector<float64_t>(variable.vlen);
		m_q.set_const(0.0);
	}
	else
	{
		REQUIRE(variable.vlen==m_q.vlen,
			"The length of variable (%d) is changed. Last time, the length of variable was %d", variable.vlen, m_q.vlen);
	}
	m_u+=proximal_weight;
	for(index_t idx=0; idx<variable.vlen; idx++)
	{
		float64_t z=variable[idx];
		if(z>0.0)
			variable[idx]=get_sparse_variable(z, m_u+m_q[idx]);
		else if(z<0.0)
			variable[idx]=get_sparse_variable(z, m_u-m_q[idx]);
		m_q[idx]+=variable[idx]-z;
	}
}

void L1PenaltyForTG::init()
{
	m_u=0;
	m_q=SGVector<float64_t>();
	SG_ADD(&m_u, "L1PenaltyForTG__m_u",
		"u in L1PenaltyForTG", MS_NOT_AVAILABLE);
	SG_ADD(&m_q, "L1PenaltyForTG__m_q",
		"q in L1PenaltyForTG", MS_NOT_AVAILABLE);
}
