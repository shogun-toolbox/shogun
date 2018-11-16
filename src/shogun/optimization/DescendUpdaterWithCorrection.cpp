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

#include <shogun/optimization/DescendUpdaterWithCorrection.h>
#include <shogun/optimization/MomentumCorrection.h>
#include <shogun/base/Parameter.h>

using namespace shogun;


DescendUpdaterWithCorrection::~DescendUpdaterWithCorrection()
{
	SG_UNREF(m_correction);
}

void DescendUpdaterWithCorrection::set_descend_correction(DescendCorrection* correction)
{
	if(m_correction != correction)
	{
		SG_REF(correction);
		SG_REF(m_correction);
		m_correction=correction;
	}
}   

void DescendUpdaterWithCorrection::update_variable(SGVector<float64_t> variable_reference,
	SGVector<float64_t> raw_negative_descend_direction, float64_t learning_rate)
{
	REQUIRE(variable_reference.vlen>0,"variable_reference must set\n");
	REQUIRE(variable_reference.vlen==raw_negative_descend_direction.vlen,
		"The length of variable_reference (%d) and the length of gradient (%d) do not match\n",
		variable_reference.vlen,raw_negative_descend_direction.vlen);

	if(m_correction)
	{
		MomentumCorrection* momentum_correction=dynamic_cast<MomentumCorrection *>(m_correction);
		if(momentum_correction)
		{
			if(!momentum_correction->is_initialized())
				momentum_correction->initialize_previous_direction(variable_reference.vlen);
		}
	}

	for(index_t idx=0; idx<variable_reference.vlen; idx++)
	{
		float64_t negative_descend_direction=get_negative_descend_direction(
			variable_reference[idx], raw_negative_descend_direction[idx], idx, learning_rate);
		if(m_correction)
		{
			DescendPair pair=m_correction->get_corrected_descend_direction(
				negative_descend_direction, idx);
			variable_reference[idx]+=pair.descend_direction;
		}
		else
		{
			variable_reference[idx]-=negative_descend_direction;
		}
	}
}

void DescendUpdaterWithCorrection::init()
{
	m_correction=NULL;
	SG_ADD((CSGObject **)&m_correction, "DescendUpdaterWithCorrection__m_correction",
		"correction in DescendUpdaterWithCorrection");
}
