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
#include <shogun/lib/SGVector.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/kernel/CustomKernel.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/statistical_testing/QuadraticTimeMMD.h>
#include <shogun/statistical_testing/internals/NextSamples.h>
#include <shogun/statistical_testing/internals/DataManager.h>
#include <shogun/statistical_testing/internals/KernelManager.h>
#include <shogun/statistical_testing/internals/mmd/FullDirect.h>

using namespace shogun;
using namespace internal;

struct CQuadraticTimeMMD::Self
{
	Self();
	index_t num_eigenvalues;
};

CQuadraticTimeMMD::Self::Self() : num_eigenvalues(0)
{
}

CQuadraticTimeMMD::CQuadraticTimeMMD() : CMMD()
{
	self = std::unique_ptr<Self>(new Self());
}

CQuadraticTimeMMD::~CQuadraticTimeMMD()
{
}

const std::function<float64_t(SGMatrix<float64_t>)> CQuadraticTimeMMD::get_direct_estimation_method() const
{
	return mmd::FullDirect();
}

const float64_t CQuadraticTimeMMD::normalize_statistic(float64_t statistic) const
{
	const DataManager& dm = get_data_manager();
	const index_t Nx = dm.num_samples_at(0);
	const index_t Ny = dm.num_samples_at(1);
	return Nx * Ny * statistic / (Nx + Ny);
}

const float64_t CQuadraticTimeMMD::normalize_variance(float64_t variance) const
{
	SG_SNOTIMPLEMENTED;
	return variance;
}

void CQuadraticTimeMMD::set_num_eigenvalues(index_t num_eigenvalues)
{
	self->num_eigenvalues = num_eigenvalues;
}

float64_t CQuadraticTimeMMD::compute_p_value(float64_t statistic)
{
	float64_t result = 0;
	switch (get_null_approximation_method())
	{
		case ENullApproximationMethod::MMD2_GAMMA:
		{
			/* fit gamma and return cdf at statistic */
			SGVector<float64_t> params = fit_null_gamma();
			result = CStatistics::gamma_cdf(statistic, params[0], params[1]);
			break;
		}
		default:
			result = CHypothesisTest::compute_p_value(statistic);
		break;
	}
	return result;
}

float64_t CQuadraticTimeMMD::compute_threshold(float64_t alpha)
{
	float64_t result = 0;
	switch (get_null_approximation_method())
	{
		case ENullApproximationMethod::MMD2_GAMMA:
		{
			/* fit gamma and return inverse cdf at alpha */
			SGVector<float64_t> params = fit_null_gamma();
			result = CStatistics::inverse_gamma_cdf(alpha, params[0], params[1]);
			break;
		}
		default:
			result = CHypothesisTest::compute_threshold(alpha);
		break;
	}
	return result;
}

SGVector<float64_t> CQuadraticTimeMMD::sample_null()
{
	if (get_null_approximation_method() == ENullApproximationMethod::MMD2_SPECTRUM)
	{
		DataManager& dm = get_data_manager();
		index_t m = dm.num_samples_at(0);
		index_t n = dm.num_samples_at(1);

		if (self->num_eigenvalues > m + n - 1)
		{
			SG_ERROR("Number of Eigenvalues too large\n");
		}

		if (self->num_eigenvalues < 1)
		{
			SG_ERROR("Number of Eigenvalues too small\n");
		}

		dm.start();
		auto next_samples = dm.next();

		SGVector<float64_t> null_samples(get_num_null_samples());
		std::fill(null_samples.vector, null_samples.vector + null_samples.vlen, 0);

		if (!next_samples.empty())
		{
			auto feats_p = next_samples[0][0];
			auto feats_q = next_samples[1][0];

			auto feats_p_q = feats_p->create_merged_copy(feats_q.get());

			CKernel *kernel = get_kernel_manager().kernel_at(0);
			kernel->init(feats_p_q, feats_p_q);
			auto precomputed = std::unique_ptr<CCustomKernel>(new CCustomKernel(kernel));
			kernel->remove_lhs_and_rhs();

			/* imaginary matrix K=[K KL; KL' L] (MATLAB notation)
			 * K is matrix for XX, L is matrix for YY, KL is XY, LK is YX
			 * works since X and Y are concatenated here */
			SGMatrix<float64_t> K = precomputed->get_kernel_matrix();

			/* center matrix K=H*K*H */
			K.center();

			/* compute eigenvalues and select num_eigenvalues largest ones */
			Eigen::Map<Eigen::MatrixXd> c_kernel_matrix(K.matrix, K.num_rows, K.num_cols);
			Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver(c_kernel_matrix);
			REQUIRE(eigen_solver.info() == Eigen::Success, "Eigendecomposition failed!\n");
			index_t max_num_eigenvalues = eigen_solver.eigenvalues().rows();

			/* finally, sample from null distribution */

#pragma omp parallel for
			for (auto i = 0; i < null_samples.vlen; ++i)
			{
				float64_t null_sample = 0;
				for (index_t j = 0; j < self->num_eigenvalues; ++j)
				{
					float64_t z_j = CMath::randn_double();
					float64_t multiple = CMath::sq(z_j);

					/* take largest EV, scale by 1/(m+n) on the fly and take abs value*/
					float64_t eigenvalue_estimate = eigen_solver.eigenvalues()[max_num_eigenvalues-1-j];
					eigenvalue_estimate /= (m + n);

					if (get_statistic_type() == EStatisticType::UNBIASED_FULL)
					{
						multiple -= 1;
					}

					null_sample += eigenvalue_estimate * multiple;
				}
				null_samples[i] = null_sample;
			}
		}

		return null_samples;
	}
	else
	{
		return CMMD::sample_null();
	}
}

SGVector<float64_t> CQuadraticTimeMMD::fit_null_gamma()
{
	DataManager& dm = get_data_manager();
	index_t m = dm.num_samples_at(0);
	index_t n = dm.num_samples_at(1);

	REQUIRE(m == n, "Only possible with equal number of samples from both distribution!\n")

	/* evtl. warn user not to use wrong statistic type */
	if (get_statistic_type() != EStatisticType::BIASED_FULL)
	{
		SG_WARNING("Note: provided statistic has to be BIASED. Please ensure that! "
		"To get rid of warning, call %s::set_statistic_type(EStatisticType::BIASED_FULL)\n", get_name());
	}

	dm.start();
	auto next_samples = dm.next();

	SGVector<float64_t> result(2);
	std::fill(result.vector, result.vector + result.vlen, 0);

	if (!next_samples.empty())
	{
		auto feats_p = next_samples[0][0];
		auto feats_q = next_samples[1][0];

		auto feats_p_q = feats_p->create_merged_copy(feats_q.get());

		CKernel *kernel = get_kernel_manager().kernel_at(0);

		/* imaginary matrix K=[K KL; KL' L] (MATLAB notation)
		 * K is matrix for XX, L is matrix for YY, KL is XY, LK is YX
		 * works since X and Y are concatenated here */
		kernel->init(feats_p_q, feats_p_q);

		/* compute mean under H0 of MMD, which is
		 * meanMMD  = 2/m * ( 1  - 1/m*sum(diag(KL))  );
		 * in MATLAB.
		 * Remove diagonals on the fly */
		float64_t mean_mmd=0;
		for (index_t i=0; i<m; ++i)
		{
			/* virtual KL matrix is in upper right corner of SHOGUN K matrix
			 * so this sums the diagonal of the matrix between X and Y*/
			mean_mmd+=kernel->kernel(i, m+i);
		}
		mean_mmd=2.0/m*(1.0-1.0/m*mean_mmd);

		/* compute variance under H0 of MMD, which is
		 * varMMD = 2/m/(m-1) * 1/m/(m-1) * sum(sum( (K + L - KL - KL').^2 ));
		 * in MATLAB, so sum up all elements */
		float64_t var_mmd=0;
		for (index_t i=0; i<m; ++i)
		{
			for (index_t j=0; j<m; ++j)
			{
				/* dont add diagonal of all pairs of imaginary kernel matrices */
				if (i==j || m+i==j || m+j==i)
					continue;

				float64_t to_add=kernel->kernel(i, j);
				to_add+=kernel->kernel(m+i, m+j);
				to_add-=kernel->kernel(i, m+j);
				to_add-=kernel->kernel(m+i, j);
				var_mmd+=CMath::pow(to_add, 2);
			}
		}

		kernel->remove_lhs_and_rhs();

		var_mmd*=2.0/m/(m-1)*1.0/m/(m-1);

		/* parameters for gamma distribution */
		float64_t a=CMath::pow(mean_mmd, 2)/var_mmd;
		float64_t b=var_mmd*m / mean_mmd;

		result[0]=a;
		result[1]=b;
	}

	return result;
}

const char* CQuadraticTimeMMD::get_name() const
{
	return "QuadraticTimeMMD";
}
