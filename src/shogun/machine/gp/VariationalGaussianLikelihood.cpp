/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Wu Lin
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

#include <shogun/lib/config.h>
#include <shogun/machine/gp/VariationalGaussianLikelihood.h>

namespace shogun
{

CVariationalGaussianLikelihood::CVariationalGaussianLikelihood()
	: CVariationalLikelihood()
{
	init();
}

void CVariationalGaussianLikelihood::init()
{
	SG_ADD(&m_mu, "mu", 
		"The mean of variational normal distribution\n",
		MS_NOT_AVAILABLE);

	SG_ADD(&m_s2, "sigma2", 
		"The variance of variational normal distribution\n",
		MS_NOT_AVAILABLE);

	SG_ADD(&m_noise_factor, "noise_factor", 
		"Correct the variance if variance is close to zero or negative\n",
		MS_NOT_AVAILABLE);
	m_noise_factor=1e-6;
}

void CVariationalGaussianLikelihood::set_noise_factor(float64_t noise_factor)
{
	REQUIRE(noise_factor>=0, "The noise_factor (%f) should be non negative\n", noise_factor);
	m_noise_factor=noise_factor;
}

bool CVariationalGaussianLikelihood::set_variational_distribution(SGVector<float64_t> mu,
	SGVector<float64_t> s2, const CLabels* lab)
{
	REQUIRE(lab, "Labels are required (lab should not be NULL)\n");
	REQUIRE((mu.vlen==s2.vlen) && (mu.vlen==lab->get_num_labels()),
		"Length of the vector of means (%d), length of the vector of "
		"variances (%d) and number of labels (%d) should be the same\n",
		mu.vlen, s2.vlen, lab->get_num_labels());

	for(index_t i = 0; i < s2.vlen; ++i)
	{
		if (!((s2[i]+m_noise_factor)>0.0)) 
			return false;
		if (!(s2[i]>0.0))
			s2[i]+=m_noise_factor;
	}

	m_mu=mu;
	m_s2=s2;
	return true;
}

}
