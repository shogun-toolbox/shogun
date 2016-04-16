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

#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/statistical_testing/LinearTimeMMD.h>
#include <shogun/statistical_testing/internals/DataManager.h>
#include <shogun/statistical_testing/internals/mmd/WithinBlockDirect.h>

using namespace shogun;
using namespace internal;

CLinearTimeMMD::CLinearTimeMMD() : CMMD()
{
}

CLinearTimeMMD::~CLinearTimeMMD()
{
}

void CLinearTimeMMD::set_num_blocks_per_burst(index_t num_blocks_per_burst)
{
	get_data_manager().set_blocksize(get_data_manager().get_min_blocksize());
	get_data_manager().set_num_blocks_per_burst(num_blocks_per_burst);
}

const std::function<float32_t(SGMatrix<float32_t>)> CLinearTimeMMD::get_direct_estimation_method() const
{
	return mmd::WithinBlockDirect();
}

const float64_t CLinearTimeMMD::normalize_statistic(float64_t statistic) const
{
	const DataManager& dm = get_data_manager();
	const index_t Nx = dm.num_samples_at(0);
	const index_t Ny = dm.num_samples_at(1);
	return CMath::sqrt(Nx * Ny / float64_t(Nx + Ny)) * statistic;
}

const float64_t CLinearTimeMMD::normalize_variance(float64_t variance) const
{
	const DataManager& dm = get_data_manager();
	const index_t Bx = dm.blocksize_at(0);
	const index_t By = dm.blocksize_at(1);
	const index_t B = Bx + By;
	if (get_statistic_type() == EStatisticType::UNBIASED_INCOMPLETE)
	{
		return variance * B * (B - 2) / 16;
	}
	return variance * Bx * By * (Bx - 1) * (By - 1) / (B - 1) / (B - 2);
}

const float64_t CLinearTimeMMD::gaussian_variance(float64_t variance) const
{
	const DataManager& dm = get_data_manager();
	const index_t Bx = dm.blocksize_at(0);
	const index_t By = dm.blocksize_at(1);
	const index_t B = Bx + By;
	if (get_statistic_type() == EStatisticType::UNBIASED_INCOMPLETE)
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
		case ENullApproximationMethod::MMD1_GAUSSIAN:
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
		case ENullApproximationMethod::MMD1_GAUSSIAN:
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
