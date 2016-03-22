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

#include <shogun/hypothesistest/BTestMMD.h>
#include <shogun/hypothesistest/internals/DataManager.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/Statistics.h>

using namespace shogun;
using namespace internal;

CBTestMMD::CBTestMMD() : CMMD<CBTestMMD>()
{
}

CBTestMMD::~CBTestMMD()
{
}

void CBTestMMD::set_blocksize(index_t blocksize)
{
	get_data_manager().set_blocksize(blocksize);
}

void CBTestMMD::set_num_blocks_per_burst(index_t num_blocks_per_burst)
{
	get_data_manager().set_num_blocks_per_burst(num_blocks_per_burst);
}

mmd::WithinBlockDirect CBTestMMD::get_direct_estimation_method()
{
	static mmd::WithinBlockDirect method;
	return method;
}

const float64_t CBTestMMD::normalize_statistic(float64_t statistic) const
{
	const DataManager& dm = get_data_manager();
	const index_t Nx = dm.num_samples_at(0);
	const index_t Ny = dm.num_samples_at(1);
	const index_t Bx = dm.blocksize_at(0);
	const index_t By = dm.blocksize_at(1);
	return Nx * Ny * statistic * CMath::sqrt((Bx + By)/float64_t(Nx + Ny)) / (Nx + Ny);
}

const float64_t CBTestMMD::normalize_variance(float64_t variance) const
{
	const DataManager& dm = get_data_manager();
	const index_t Bx = dm.blocksize_at(0);
	const index_t By = dm.blocksize_at(1);
	return variance * CMath::sq(Bx * By / float64_t(Bx + By));
}

float64_t CBTestMMD::compute_p_value(float64_t statistic)
{
	float64_t result = 0;
	switch (get_null_approximation_method())
	{
		case N_METHOD::MMD1_GAUSSIAN:
		{
			float64_t sigma_sq = compute_variance();
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

float64_t CBTestMMD::compute_threshold(float64_t alpha)
{
	float64_t result = 0;
	switch (get_null_approximation_method())
	{
		case N_METHOD::MMD1_GAUSSIAN:
		{
			float64_t sigma_sq = compute_variance();
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

const char* CBTestMMD::get_name() const
{
	return "BTestMMD";
}
