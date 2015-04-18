/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2015 Esben Soerig
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
 */

#include <shogun/kernel/PeriodicKernel.h>

using namespace shogun;

CPeriodicKernel::CPeriodicKernel() : CDotKernel()
{
	init();
}

CPeriodicKernel::CPeriodicKernel(float64_t ls, float64_t p, int32_t s) : CDotKernel(s)
{
	init();
	set_length_scale(ls);
	set_period(p);
}

CPeriodicKernel::CPeriodicKernel(CDotFeatures* l, CDotFeatures* r, float64_t ls,
	float64_t p, int32_t s) : CDotKernel(s)
{
	init();
	set_length_scale(ls);
	set_period(p);
	init(l,r);
}

void CPeriodicKernel::precompute_squared_helper(SGVector<float64_t>& buf,
	CDotFeatures* df)
{
	int32_t num_vec=df->get_num_vectors();
	buf=SGVector<float64_t>(num_vec);

	for (int32_t i=0; i<num_vec; i++)
		buf[i]=df->dot(i,df, i);
}

bool CPeriodicKernel::init(CFeatures* l, CFeatures* r)
{
	CDotKernel::init(l, r);
	precompute_squared();
	return init_normalizer();
}

float64_t CPeriodicKernel::compute(int32_t idx_a, int32_t idx_b)
{
	/* Periodic kernel defined as by David Duvenaud in
	 http://mlg.eng.cam.ac.uk/duvenaud/cookbook/index.html

	 k({\bf x},{\bf x'})= exp(-\frac{2sin^2(|{\bf x}-{\bf x'}|/p)}{l^2})
	*/
	float64_t sin_term=CMath::sin(M_PI*distance(idx_a,idx_b)/m_period);
	return CMath::exp(-2*CMath::pow(sin_term, 2)/CMath::pow(m_length_scale,2));
}

void CPeriodicKernel::precompute_squared()
{
	if (!lhs || !rhs)
		return;

	CDotFeatures* dotlhs=dynamic_cast<CDotFeatures*>(lhs);
	REQUIRE(dotlhs!=NULL, "Left-hand-side features must be of type CDotFeatures\n")

	precompute_squared_helper(m_sq_lhs, dotlhs);

	if (lhs==rhs)
		m_sq_rhs=m_sq_lhs;
	else
	{
		CDotFeatures *dotrhs=dynamic_cast<CDotFeatures*>(rhs);
		REQUIRE(dotrhs!=NULL, "Left-hand-side features must be of type CDotFeatures\n")

		precompute_squared_helper(m_sq_rhs, dotrhs);
	}
}

SGMatrix<float64_t> CPeriodicKernel::get_parameter_gradient(
	const TParameter* param, index_t index)
{
	REQUIRE(lhs, "Left-hand-side features not set!\n")
	REQUIRE(rhs, "Right-hand-side features not set!\n")

	if (!strcmp(param->m_name, "length_scale"))
	{
		SGMatrix<float64_t> derivative=SGMatrix<float64_t>(num_lhs, num_rhs);

		for (int j=0; j<num_lhs; j++)
		{
			for (int k=0; k<num_rhs; k++)
			{
				float64_t dist=distance(j,k);
				float64_t trig_arg=M_PI*dist/m_period;
				float64_t original=compute(j, k);

				derivative(j,k)=original*4.0*pow(sin(trig_arg),2)/pow(m_length_scale,3);
			}
		}

		return derivative;
	}
	else if (!strcmp(param->m_name, "period"))
	{
		SGMatrix<float64_t> derivative=SGMatrix<float64_t>(num_lhs, num_rhs);

		for (int j=0; j<num_lhs; j++)
		{
			for (int k=0; k<num_rhs; k++)
			{
				float64_t dist=distance(j,k);
				float64_t trig_arg=M_PI*dist/m_period;
				float64_t original=compute(j, k);

				derivative(j,k)=original*4.0*M_PI*dist*cos(trig_arg)*sin(trig_arg)/pow(m_period*m_length_scale,2);
			}
		}
		return derivative;
	}
	else
	{
		SG_ERROR("Can't compute derivative wrt %s parameter\n", param->m_name);
		return SGMatrix<float64_t>();
	}
}

void CPeriodicKernel::init()
{
	set_length_scale(1.0);
	set_period(1.0);

	SG_ADD(&m_length_scale, "length_scale",
		"Kernel length scale", MS_AVAILABLE, GRADIENT_AVAILABLE);
	SG_ADD(&m_period, "period",
		"Kernel period", MS_AVAILABLE, GRADIENT_AVAILABLE);
	SG_ADD(&m_sq_lhs, "sq_lhs",
		"Vector of dot products of each left-hand-side vector with itself.", MS_NOT_AVAILABLE);
	SG_ADD(&m_sq_rhs, "sq_rhs",
		"Vector of dot products of each right-hand-side vector with itself.", MS_NOT_AVAILABLE);
}

float64_t CPeriodicKernel::distance(int32_t idx_a, int32_t idx_b)
{
	return sqrt(m_sq_lhs[idx_a]+m_sq_rhs[idx_b]-2*CDotKernel::compute(idx_a,idx_b));
}
