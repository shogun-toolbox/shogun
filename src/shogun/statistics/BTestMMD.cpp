/*
 * Copyright (c) The Shogun Machine Learning Toolbox
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

#include <shogun/statistics/BTestMMD.h>
#include <shogun/features/Features.h>
#include <shogun/features/streaming/StreamingFeatures.h>

using namespace shogun;

CBTestMMD::CBTestMMD() : CStreamingMMD()
{
}

CBTestMMD::CBTestMMD(CKernel* kernel, CStreamingFeatures* p,
		CStreamingFeatures* q, index_t m, index_t blocksize)
	: CStreamingMMD(kernel, p, q, m, m)
{
	set_blocksize(blocksize);
}

CBTestMMD::CBTestMMD(CKernel* kernel, CStreamingFeatures* p,
		CStreamingFeatures* q, index_t m, index_t n, index_t blocksize)
	: CStreamingMMD(kernel, p, q, m, n)
{
	set_blocksize(blocksize);
}

CBTestMMD::~CBTestMMD()
{
}

float64_t CBTestMMD::compute_statistic_normalizing_constant()
{
	index_t n=m_m+m_n;
	return 1.0/n*m_m*m_n*CMath::sqrt(m_blocksize/float64_t(n));
}

float64_t CBTestMMD::compute_variance_normalizing_constant()
{
	index_t Bx=m_blocksize_p;
	index_t By=m_blocksize_q;

	return CMath::sq(1.0/(Bx+By)*Bx*By);
}

float64_t CBTestMMD::compute_gaussian_variance(float64_t variance)
{
	return variance;
}

