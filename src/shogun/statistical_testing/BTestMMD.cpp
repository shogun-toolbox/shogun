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
#include <shogun/distance/CustomDistance.h>
#include <shogun/statistical_testing/BTestMMD.h>
#include <shogun/statistical_testing/internals/DataManager.h>
#include <shogun/statistical_testing/internals/mmd/WithinBlockDirect.h>

using namespace shogun;
using namespace internal;

CBTestMMD::CBTestMMD() : CMMD()
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

const float64_t CBTestMMD::normalize_statistic(float64_t statistic) const
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
