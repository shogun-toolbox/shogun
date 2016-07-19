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

#include <shogun/optimization/MomentumCorrection.h>
#include <shogun/base/Parameter.h>
using namespace shogun;

void MomentumCorrection::initialize_previous_direction(index_t len)
{
	REQUIRE(len>0, "The length (%d) must be positive\n", len);
	m_previous_descend_direction=SGVector<float64_t>(len);
	m_previous_descend_direction.set_const(0.0);
}

float64_t MomentumCorrection::get_previous_descend_direction(index_t idx)
{
	REQUIRE(idx>=0 && idx<m_previous_descend_direction.vlen,
		"Index (%d) is invalid\n", idx);
	return m_previous_descend_direction[idx];
}

void MomentumCorrection::init()
{
	m_previous_descend_direction=SGVector<float64_t>();
	SG_ADD(&m_previous_descend_direction, "MomentumCorrection__m_previous_descend_direction",
		"previous_descend_direction in MomentumCorrection", MS_NOT_AVAILABLE);
}
