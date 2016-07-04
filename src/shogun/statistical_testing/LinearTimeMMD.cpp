/*
 * Restructuring Shogun's statistical hypothesis testing framework.
 * Copyright (C) 2016  Soumyajit De
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <shogun/io/SGIO.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/distance/CustomDistance.h>
#include <shogun/statistical_testing/LinearTimeMMD.h>
#include <shogun/statistical_testing/internals/DataManager.h>
#include <shogun/statistical_testing/internals/mmd/WithinBlockDirect.h>

using namespace shogun;
using namespace internal;

CLinearTimeMMD::CLinearTimeMMD() : CMMD()
{
}

CLinearTimeMMD::CLinearTimeMMD(CFeatures* samples_from_p, CFeatures* samples_from_q) : CMMD()
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

const float64_t CLinearTimeMMD::normalize_statistic(float64_t statistic) const
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
	if (get_statistic_type() == EStatisticType::ST_UNBIASED_INCOMPLETE)
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
	if (get_statistic_type() == EStatisticType::ST_UNBIASED_INCOMPLETE)
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
		case ENullApproximationMethod::NAM_MMD1_GAUSSIAN:
		{
			float64_t sigma_sq = gaussian_variance(compute_variance());
			float64_t std_dev = CMath::sqrt(sigma_sq);
			result = 1.0 - CStatistics::normal_cdf(statistic, std_dev);
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
		case ENullApproximationMethod::NAM_MMD1_GAUSSIAN:
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
