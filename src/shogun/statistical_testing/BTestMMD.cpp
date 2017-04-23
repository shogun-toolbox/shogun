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

#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/distance/CustomDistance.h>
#include <shogun/statistical_testing/BTestMMD.h>
#include <shogun/statistical_testing/TestEnums.h>
#include <shogun/statistical_testing/internals/DataManager.h>
#include <shogun/statistical_testing/internals/mmd/WithinBlockDirect.h>

using namespace shogun;
using namespace internal;

CBTestMMD::CBTestMMD() : CStreamingMMD()
{
}

CBTestMMD::~CBTestMMD()
{
}

void CBTestMMD::set_blocksize(index_t blocksize)
{
	get_data_mgr().set_blocksize(blocksize);
}

void CBTestMMD::set_num_blocks_per_burst(index_t num_blocks_per_burst)
{
	get_data_mgr().set_num_blocks_per_burst(num_blocks_per_burst);
}

const std::function<float32_t(SGMatrix<float32_t>)> CBTestMMD::get_direct_estimation_method() const
{
	return mmd::WithinBlockDirect();
}

float64_t CBTestMMD::normalize_statistic(float64_t statistic) const
{
	const DataManager& data_mgr=get_data_mgr();
	const index_t Nx=data_mgr.num_samples_at(0);
	const index_t Ny=data_mgr.num_samples_at(1);
	const index_t Bx=data_mgr.blocksize_at(0);
	const index_t By=data_mgr.blocksize_at(1);
	return Nx*Ny*statistic*CMath::sqrt((Bx+By)/float64_t(Nx+Ny))/(Nx+Ny);
}

const float64_t CBTestMMD::normalize_variance(float64_t variance) const
{
	const DataManager& data_mgr=get_data_mgr();
	const index_t Bx=data_mgr.blocksize_at(0);
	const index_t By=data_mgr.blocksize_at(1);
	return variance*CMath::sq(Bx*By/float64_t(Bx+By));
}

float64_t CBTestMMD::compute_p_value(float64_t statistic)
{
	float64_t result=0;
	switch (get_null_approximation_method())
	{
		case NAM_MMD1_GAUSSIAN:
		{
			float64_t sigma_sq=compute_variance();
			float64_t std_dev=CMath::sqrt(sigma_sq);
			result=1.0-CStatistics::normal_cdf(statistic, std_dev);
			break;
		}
		default:
		{
			result=CHypothesisTest::compute_p_value(statistic);
			break;
		}
	}
	return result;
}

float64_t CBTestMMD::compute_threshold(float64_t alpha)
{
	float64_t result=0;
	switch (get_null_approximation_method())
	{
		case NAM_MMD1_GAUSSIAN:
		{
			float64_t sigma_sq=compute_variance();
			float64_t std_dev=CMath::sqrt(sigma_sq);
			result=1.0-CStatistics::inverse_normal_cdf(1-alpha, 0, std_dev);
			break;
		}
		default:
		{
			result=CHypothesisTest::compute_threshold(alpha);
			break;
		}
	}
	return result;
}

const char* CBTestMMD::get_name() const
{
	return "BTestMMD";
}
