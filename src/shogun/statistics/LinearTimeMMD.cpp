/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2012-2013 Heiko Strathmann
 * Written (w) 2014 Soumyajit De
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

#include <shogun/statistics/LinearTimeMMD.h>
#include <shogun/features/Features.h>
#include <shogun/features/streaming/StreamingFeatures.h>

using namespace shogun;

CLinearTimeMMD::CLinearTimeMMD() : CStreamingMMD()
{
}

CLinearTimeMMD::CLinearTimeMMD(CKernel* kernel, CStreamingFeatures* p,
		CStreamingFeatures* q, index_t m, index_t blocksize)
	: CStreamingMMD(kernel, p, q, m, m)
{
	set_blocksize(blocksize);
}

CLinearTimeMMD::CLinearTimeMMD(CKernel* kernel, CStreamingFeatures* p,
		CStreamingFeatures* q, index_t m, index_t n, index_t blocksize)
	: CStreamingMMD(kernel, p, q, m, n)
{
	set_blocksize(blocksize);
}

CLinearTimeMMD::~CLinearTimeMMD()
{
}

float64_t CLinearTimeMMD::compute_statistic_normalizing_constant()
{
	return CMath::sqrt(1.0/(m_m+m_n)*m_m*m_n);
}

float64_t CLinearTimeMMD::compute_variance_normalizing_constant()
{
	index_t B=m_blocksize;
	index_t Bx=m_blocksize_p;
	index_t By=m_blocksize_q;

	switch (m_statistic_type)
	{
		case S_UNBIASED:
			return 1.0/(B-1)/(B-2)*Bx*By*(Bx-1)*(By-1);
		case S_INCOMPLETE:
			return 1.0/16.0*B*(B-2);
		default:
			SG_ERROR("Unknown statistic type (%d) is used!\n", m_statistic_type);
			return 0;
	}
}

float64_t CLinearTimeMMD::compute_gaussian_variance(float64_t variance)
{
	index_t B=m_blocksize;
	index_t Bx=m_blocksize_p;
	index_t By=m_blocksize_q;

	switch (m_statistic_type)
	{
		case S_UNBIASED:
			return variance/B/(Bx-1)/(By-1)*(B-1)*(B-2);
		case S_INCOMPLETE:
			return variance*4.0/(B-2);
		default:
			SG_ERROR("Unknown statistic type (%d) is used!\n", m_statistic_type);
			return 0;
	}
}

