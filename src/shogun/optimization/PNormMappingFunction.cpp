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

#include <shogun/optimization/PNormMappingFunction.h>
#include <shogun/mathematics/Math.h>
#include <shogun/base/Parameter.h>
using namespace shogun;

void PNormMappingFunction::set_norm(float64_t p)
{
	if(p<2.0)
	{
		SG_SWARNING("The norm (%f) should not be less than 2.0 and we use p=2.0 in this case\n", p);
		m_p=2.0;
	}
	else
		m_p=p;
}

SGVector<float64_t> PNormMappingFunction::get_dual_variable(SGVector<float64_t> variable)
{
	SGVector<float64_t> dual_variable(variable.vlen);
	float64_t q=1.0/(1.0-1.0/m_p);
	projection(variable, dual_variable, q);
	return dual_variable;
}

void PNormMappingFunction::update_variable(SGVector<float64_t> variable, SGVector<float64_t> dual_variable)
{
	projection(dual_variable, variable, m_p);
}

void PNormMappingFunction::projection(SGVector<float64_t> input, SGVector<float64_t> output, float64_t degree)
{
	REQUIRE(input.vlen==output.vlen,"The lenght (%d) of input and the length (%d) of output are diffent\n",
		input.vlen, output.vlen);
	float64_t scale=0.0;
	for(index_t idx=0; idx<input.vlen; idx++)
	{
		scale += CMath::pow(CMath::abs(input[idx]),degree);
		if (input[idx] >= 0.0)
			output[idx]=CMath::pow(input[idx],degree-1);
		else
			output[idx]=-CMath::pow(-input[idx],degree-1);
	}
	scale=CMath::pow(scale,1.0-2.0/degree);
	for(index_t idx=0; idx<input.vlen; idx++)
		output[idx]/=scale;
}

void PNormMappingFunction::init()
{
	m_p=2.0;
	SG_ADD(&m_p, "PNormMappingFunction__m_p",
		"p in PNormMappingFunction", MS_NOT_AVAILABLE);
}
