/** The Shogun Machine Learning Toolbox
 *  Copyright (c) 2014, The Shogun-Team
 * All rights reserved.
 *
 * Distributed under the BSD 2-clause license:
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
 * SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <shogun/kernel/DiffusionKernel.h>
#include <shogun/features/DenseFeatures.h>

using namespace shogun;

CDiffusionKernel::CDiffusionKernel() : CDotKernel()
{
	init();
}

CDiffusionKernel::CDiffusionKernel(SGVector<float64_t> betas,
		SGVector<index_t> alphabet_sizes) : CDotKernel()
{
	init();

	m_betas=betas;
	m_alphabet_sizes=alphabet_sizes;

	precompute_decay_numbers();
}

CDiffusionKernel::~CDiffusionKernel()
{
}

bool CDiffusionKernel::init(CFeatures* l, CFeatures* r)
{
	REQUIRE(dynamic_cast<CDenseFeatures<float64_t>*>(l), "Only works for 64bit "
			"dense features for now. LHS is not.");
	REQUIRE(dynamic_cast<CDenseFeatures<float64_t>*>(r), "Only works for 64bit "
			"dense features for now. RHS is not.");

	return CKernel::init(l, r);
}

float64_t CDiffusionKernel::compute(int32_t idx_a, int32_t idx_b)
{
	/* evil hack (a check is done in init) */
	SGVector<float64_t> l=
			((CDenseFeatures<float64_t>*) lhs)->get_feature_vector(idx_a);
	SGVector<float64_t> r=
			((CDenseFeatures<float64_t>*) rhs)->get_feature_vector(idx_b);

	float64_t result=1.0;
	for (index_t i=0; i<l.vlen; ++i)
	{
		/* another evil hack to ensure things are "equal" in floating points */
		if (CMath::abs(l[i]-r[i])>1e-5)
			result*=m_decay_numbers[i];
	}

	return result;
}

void CDiffusionKernel::init()
{
}

void CDiffusionKernel::precompute_decay_numbers()
{
	REQUIRE(m_betas.vlen==m_alphabet_sizes.vlen, "Beta vector length (%d) and "
			"alphabet sizes vector (%d) must be equal.", m_betas.vlen,
			m_alphabet_sizes.vlen);

	/* reallocate decay number memory if necessary, otherwise overwrite */
	if (m_decay_numbers.vlen!=m_betas.vlen)
		m_decay_numbers=SGVector<float64_t>(m_betas.vlen);

	/* pre-compute decay numbers */
	for (index_t i=0; i<m_betas.vlen; ++i)
	{
		float64_t eAb=CMath::exp(-m_alphabet_sizes[i]*m_betas[i]);
		float64_t a=1-eAb;
		float64_t b=1+(m_alphabet_sizes[i]-1)*eAb;
		m_decay_numbers[i]=a/b;
	}
}

void CDiffusionKernel::set_betas(SGVector<float64_t> betas)
{
	m_betas=betas;
	precompute_decay_numbers();
}
