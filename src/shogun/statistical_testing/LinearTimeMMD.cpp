/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2012 - 2013 Heiko Strathmann
 * Written (w) 2014 - 2017 Soumyajit De
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

#include <shogun/io/SGIO.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/distance/CustomDistance.h>
#include <shogun/statistical_testing/LinearTimeMMD.h>
#include <shogun/statistical_testing/TestEnums.h>
#include <shogun/statistical_testing/internals/DataManager.h>
#include <shogun/statistical_testing/internals/mmd/WithinBlockDirect.h>

using namespace shogun;
using namespace internal;

CLinearTimeMMD::CLinearTimeMMD() : CStreamingMMD()
{
}

CLinearTimeMMD::CLinearTimeMMD(CFeatures* samples_from_p, CFeatures* samples_from_q) : CStreamingMMD()
{
	set_p(samples_from_p);
	set_q(samples_from_q);
}

CLinearTimeMMD::~CLinearTimeMMD()
{
}

void CLinearTimeMMD::set_num_blocks_per_burst(index_t num_blocks_per_burst)
{
	auto& data_mgr=get_data_mgr();
	auto min_blocksize=data_mgr.get_min_blocksize();
	if (min_blocksize==2)
	{
		// only possible when number of samples from both the distributions are the same
		auto N=data_mgr.num_samples_at(0);
		for (auto i=2; i<N; ++i)
		{
			if (N%i==0)
			{
				min_blocksize=i*2;
				break;
			}
		}
	}
	data_mgr.set_blocksize(min_blocksize);
	data_mgr.set_num_blocks_per_burst(num_blocks_per_burst);
	SG_SDEBUG("Block contains %d and %d samples, from P and Q respectively!\n", data_mgr.blocksize_at(0), data_mgr.blocksize_at(1));
}

const std::function<float32_t(SGMatrix<float32_t>)> CLinearTimeMMD::get_direct_estimation_method() const
{
	return mmd::WithinBlockDirect();
}

float64_t CLinearTimeMMD::normalize_statistic(float64_t statistic) const
{
	const DataManager& data_mgr = get_data_mgr();
	const index_t Nx = data_mgr.num_samples_at(0);
	const index_t Ny = data_mgr.num_samples_at(1);
	return CMath::sqrt(Nx * Ny / float64_t(Nx + Ny)) * statistic;
}

const float64_t CLinearTimeMMD::normalize_variance(float64_t variance) const
{
	const DataManager& data_mgr = get_data_mgr();
	const index_t Bx = data_mgr.blocksize_at(0);
	const index_t By = data_mgr.blocksize_at(1);
	const index_t B = Bx + By;
	if (get_statistic_type() == ST_UNBIASED_INCOMPLETE)
	{
		return variance * B * (B - 2) / 16;
	}
	return variance * Bx * By * (Bx - 1) * (By - 1) / (B - 1) / (B - 2);
}

const float64_t CLinearTimeMMD::gaussian_variance(float64_t variance) const
{
	const DataManager& data_mgr = get_data_mgr();
	const index_t Bx = data_mgr.blocksize_at(0);
	const index_t By = data_mgr.blocksize_at(1);
	const index_t B = Bx + By;
	if (get_statistic_type() == ST_UNBIASED_INCOMPLETE)
	{
		return variance * 4 / (B - 2);
	}
	return variance * (B - 1) * (B - 2) / (Bx - 1) / (By - 1) / B;
}

float64_t CLinearTimeMMD::compute_p_value(float64_t statistic)
{
	float64_t result = 0;
	switch (get_null_approximation_method())
	{
		case NAM_MMD1_GAUSSIAN:
		{
			float64_t sigma_sq = gaussian_variance(compute_variance());
			float64_t std_dev = CMath::sqrt(sigma_sq);
			result = 1.0 - CStatistics::normal_cdf(statistic, std_dev);
			break;
		}
		case NAM_PERMUTATION:
		{
			SG_SERROR("Null approximation via permutation does not make sense "
					"for linear time MMD. Use the Gaussian approximation instead.\n");
			break;
		}
		default:
		{
			result = CHypothesisTest::compute_p_value(statistic);
			break;
		}
	}
	return result;
}

float64_t CLinearTimeMMD::compute_threshold(float64_t alpha)
{
	float64_t result = 0;
	switch (get_null_approximation_method())
	{
		case NAM_MMD1_GAUSSIAN:
		{
			float64_t sigma_sq = gaussian_variance(compute_variance());
			float64_t std_dev = CMath::sqrt(sigma_sq);
			result = 1.0 - CStatistics::inverse_normal_cdf(1 - alpha, 0, std_dev);
			break;
		}
		default:
		{
			result = CHypothesisTest::compute_threshold(alpha);
			break;
		}
	}
	return result;
}

const char* CLinearTimeMMD::get_name() const
{
	return "LinearTimeMMD";
}
