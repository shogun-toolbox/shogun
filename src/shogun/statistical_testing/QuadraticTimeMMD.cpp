/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2012 - 2013 Heiko Strathmann
 * Written (w) 2014 - 2016 Soumyajit De
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

#include <algorithm>
#include <shogun/io/SGIO.h>
#include <shogun/lib/SGVector.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/kernel/CustomKernel.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/distance/EuclideanDistance.h>
#include <shogun/distance/CustomDistance.h>
#include <shogun/statistical_testing/QuadraticTimeMMD.h>
#include <shogun/statistical_testing/internals/FeaturesUtil.h>
#include <shogun/statistical_testing/internals/NextSamples.h>
#include <shogun/statistical_testing/internals/DataManager.h>
#include <shogun/statistical_testing/internals/KernelManager.h>
#include <shogun/statistical_testing/internals/ComputationManager.h>
#include <shogun/statistical_testing/internals/mmd/BiasedFull.h>
#include <shogun/statistical_testing/internals/mmd/UnbiasedFull.h>
#include <shogun/statistical_testing/internals/mmd/UnbiasedIncomplete.h>
#include <shogun/statistical_testing/internals/mmd/FullDirect.h>
#include <shogun/statistical_testing/internals/mmd/WithinBlockPermutationBatch.h>

using namespace shogun;
using namespace internal;
using namespace mmd;

struct CQuadraticTimeMMD::Self
{
	Self(CQuadraticTimeMMD&);

	SGMatrix<float32_t> get_kernel_matrix();
	std::pair<float64_t, float64_t> compute_statistic_variance();
	SGVector<float64_t> sample_null();

	void create_statistic_job();
	void create_variance_job();
	void create_computation_jobs();

	void compute_jobs(ComputationManager&) const;

	CQuadraticTimeMMD& owner;
	index_t num_eigenvalues;

	std::function<float32_t(const SGMatrix<float32_t>&)> statistic_job;
	std::function<float32_t(const SGMatrix<float32_t>&)> variance_job;
};

CQuadraticTimeMMD::Self::Self(CQuadraticTimeMMD& mmd) : owner(mmd), num_eigenvalues(10),
	statistic_job(nullptr), variance_job(nullptr)
{
}

void CQuadraticTimeMMD::Self::create_computation_jobs()
{
	SG_SDEBUG("Entering\n");
	create_statistic_job();
	create_variance_job();
	SG_SDEBUG("Leaving\n");
}

void CQuadraticTimeMMD::Self::create_statistic_job()
{
	SG_SDEBUG("Entering\n");
	const DataManager& dm=owner.get_data_manager();
	auto Nx=dm.num_samples_at(0);
	switch (owner.get_statistic_type())
	{
		case ST_UNBIASED_FULL:
			statistic_job=UnbiasedFull(Nx);
			break;
		case ST_UNBIASED_INCOMPLETE:
			statistic_job=UnbiasedIncomplete(Nx);
			break;
		case ST_BIASED_FULL:
			statistic_job=BiasedFull(Nx);
			break;
		default : break;
	};
	SG_SDEBUG("Leaving\n");
}

void CQuadraticTimeMMD::Self::create_variance_job()
{
	SG_SDEBUG("Entering\n");
	switch (owner.get_variance_estimation_method())
	{
		case VEM_DIRECT:
			variance_job=owner.get_direct_estimation_method();
			break;
		case VEM_PERMUTATION:
			SG_SERROR("Permutation method is not allowed with Quadratic Time MMD!\n");
			break;
		default : break;
	};
	SG_SDEBUG("Leaving\n");
}

void CQuadraticTimeMMD::Self::compute_jobs(ComputationManager& cm) const
{
	SG_SDEBUG("Entering\n");
	if (owner.use_gpu())
		cm.use_gpu().compute_task_parallel_jobs();
	else
		cm.use_cpu().compute_task_parallel_jobs();
	SG_SDEBUG("Leaving\n");
}

SGMatrix<float32_t> CQuadraticTimeMMD::Self::get_kernel_matrix()
{
	SG_SDEBUG("Entering\n");
	const KernelManager& km=owner.get_kernel_manager();
	auto kernel=km.kernel_at(0);
	REQUIRE(kernel!=nullptr, "Kernel is not set!\n");

	SGMatrix<float32_t> kernel_matrix;

	// check if precomputed kernel is given, no need to do anything in that case
	// otherwise, init kernel with data and precompute kernel matrix
	if (kernel->get_kernel_type()==K_CUSTOM)
	{
		auto precomputed_kernel=dynamic_cast<CCustomKernel*>(kernel);
		ASSERT(precomputed_kernel!=nullptr);
		kernel_matrix=precomputed_kernel->get_float32_kernel_matrix();
	}
	else
	{
		DataManager& dm=owner.get_data_manager();

		// using data manager next() API in order to make it work with
		// streaming samples as well.
		dm.start();
		auto samples=dm.next();
		if (!samples.empty())
		{
			dm.end();

			// use 0th block from each distribution (since there is only one block
			// for quadratic time MMD
			CFeatures *samples_p=samples[0][0].get();
			CFeatures *samples_q=samples[1][0].get();
			auto samples_p_and_q=FeaturesUtil::create_merged_copy(samples_p, samples_q);
			samples.clear();
			kernel->init(samples_p_and_q, samples_p_and_q);
			try
			{
				owner.get_kernel_manager().precompute_kernel_at(0);
			}
			catch (ShogunException e)
			{
				SG_SERROR("%s, Data is too large! Computing kernel matrix was not possible!\n", e.get_exception_string());
			}
			kernel->remove_lhs_and_rhs();
			auto precomputed_kernel=dynamic_cast<CCustomKernel*>(km.kernel_at(0));
			ASSERT(precomputed_kernel!=nullptr);
			kernel_matrix=precomputed_kernel->get_float32_kernel_matrix();
		}
		else
		{
			dm.end();
			SG_SERROR("Could not fetch samples!\n");
		}
	}

	SG_SDEBUG("Leaving\n");
	return kernel_matrix;
}

std::pair<float64_t, float64_t> CQuadraticTimeMMD::Self::compute_statistic_variance()
{
	SG_SDEBUG("Entering\n");
	SGMatrix<float32_t> kernel_matrix=get_kernel_matrix();

	ComputationManager cm;
	create_computation_jobs();
	cm.enqueue_job(statistic_job);
	cm.enqueue_job(variance_job);

	cm.num_data(1);
	cm.data(0)=kernel_matrix;

	compute_jobs(cm);
	auto mmd=cm.result(0);
	auto var=cm.result(1);
	float64_t statistic=mmd[0];
	float64_t variance=var[0];
	cm.done();

	SG_SDEBUG("statistic=%f [un-normalized]\n", statistic);
	statistic=owner.normalize_statistic(statistic);
	SG_SDEBUG("statistic=%f [normalized]\n", statistic);
	SG_SDEBUG("variance=%f [normalized]\n", variance);

	SG_SDEBUG("Leaving\n");
	return std::make_pair(statistic, variance);
}

SGVector<float64_t> CQuadraticTimeMMD::Self::sample_null()
{
	SG_SDEBUG("Entering\n");
	SGMatrix<float32_t> kernel_matrix=get_kernel_matrix();

	const DataManager& dm=owner.get_data_manager();
	auto Nx=dm.num_samples_at(0);
	auto Ny=dm.num_samples_at(1);

	WithinBlockPermutationBatch compute(Nx, Ny, owner.get_num_null_samples(), owner.get_statistic_type());
	SGVector<float32_t> result=compute(kernel_matrix);
	SGVector<float64_t> null_samples(result.vlen);
	for (auto i=0; i<result.vlen; ++i)
		null_samples[i]=owner.normalize_statistic(result[i]);
	SG_SDEBUG("Leaving\n");
	return null_samples;
}

CQuadraticTimeMMD::CQuadraticTimeMMD() : CMMD()
{
	self=std::unique_ptr<Self>(new Self(*this));
}

CQuadraticTimeMMD::CQuadraticTimeMMD(CFeatures* samples_from_p, CFeatures* samples_from_q) : CMMD()
{
	self=std::unique_ptr<Self>(new Self(*this));
	set_p(samples_from_p);
	set_q(samples_from_q);
}

CQuadraticTimeMMD::~CQuadraticTimeMMD()
{
	get_kernel_manager().restore_kernel_at(0);
}

const std::function<float32_t(SGMatrix<float32_t>)> CQuadraticTimeMMD::get_direct_estimation_method() const
{
	return FullDirect();
}

const float64_t CQuadraticTimeMMD::normalize_statistic(float64_t statistic) const
{
	const DataManager& dm=get_data_manager();
	const index_t Nx=dm.num_samples_at(0);
	const index_t Ny=dm.num_samples_at(1);
	return Nx*Ny*statistic/(Nx+Ny);
}

const float64_t CQuadraticTimeMMD::normalize_variance(float64_t variance) const
{
	SG_SNOTIMPLEMENTED;
	return variance;
}

void CQuadraticTimeMMD::spectrum_set_num_eigenvalues(index_t num_eigenvalues)
{
	self->num_eigenvalues=num_eigenvalues;
}

float64_t CQuadraticTimeMMD::compute_statistic()
{
	return self->compute_statistic_variance().first;
}

float64_t CQuadraticTimeMMD::compute_variance()
{
	return self->compute_statistic_variance().second;
}

float64_t CQuadraticTimeMMD::compute_p_value(float64_t statistic)
{
	SG_DEBUG("Entering\n");
	float64_t result=0;
	switch (get_null_approximation_method())
	{
		case NAM_MMD2_GAMMA:
		{
			SGVector<float64_t> params=gamma_fit_null();
			result=CStatistics::gamma_cdf(statistic, params[0], params[1]);
			break;
		}
		default:
			result=CHypothesisTest::compute_p_value(statistic);
		break;
	}
	SG_DEBUG("Leaving\n");
	return result;
}

float64_t CQuadraticTimeMMD::compute_threshold(float64_t alpha)
{
	SG_DEBUG("Entering\n");
	float64_t result=0;
	switch (get_null_approximation_method())
	{
		case NAM_MMD2_GAMMA:
		{
			SGVector<float64_t> params=gamma_fit_null();
			result=CStatistics::gamma_inverse_cdf(alpha, params[0], params[1]);
			break;
		}
		default:
			result=CHypothesisTest::compute_threshold(alpha);
			break;
	}
	SG_DEBUG("Leaving\n");
	return result;
}

SGVector<float64_t> CQuadraticTimeMMD::sample_null()
{
	SG_DEBUG("Entering\n");
	SGVector<float64_t> null_samples;
	switch (get_null_approximation_method())
	{
		case NAM_MMD2_SPECTRUM:
			null_samples=spectrum_sample_null();
			break;
		default:
			null_samples=self->sample_null();
			break;
		}

	SG_DEBUG("Leaving\n");
	return null_samples;
}

SGVector<float64_t> CQuadraticTimeMMD::gamma_fit_null()
{
	SG_DEBUG("Entering\n");
	DataManager& dm=get_data_manager();
	index_t m=dm.num_samples_at(0);
	index_t n=dm.num_samples_at(1);

	REQUIRE(m==n, "Number of samples from p (%d) and q (%d) must be equal.\n", n, m)

	/* evtl. warn user not to use wrong statistic type */
	if (get_statistic_type()!=ST_BIASED_FULL)
	{
		SG_WARNING("Note: provided statistic has to be BIASED. Please ensure that! "
		"To get rid of warning, call %s::set_statistic_type(EStatisticType::BIASED_FULL)\n", get_name());
	}

	SGVector<float64_t> result(2);
	std::fill(result.vector, result.vector+result.vlen, 0);

	/* imaginary matrix K=[K KL; KL' L] (MATLAB notation)
	 * K is matrix for XX, L is matrix for YY, KL is XY, LK is YX
	 * works since X and Y are concatenated here */
	SGMatrix<float32_t> kernel_matrix=self->get_kernel_matrix();

	/* compute mean under H0 of MMD, which is
	 * meanMMD =2/m * ( 1  - 1/m*sum(diag(KL))  );
	 * in MATLAB.
	 * Remove diagonals on the fly */
	float64_t mean_mmd=0;
	for (index_t i=0; i<m; ++i)
	{
		/* virtual KL matrix is in upper right corner of SHOGUN K matrix
		 * so this sums the diagonal of the matrix between X and Y*/
		mean_mmd+=kernel_matrix(i, m+i);
	}
	mean_mmd=2.0/m*(1.0-1.0/m*mean_mmd);

	/* compute variance under H0 of MMD, which is
	 * varMMD=2/m/(m-1) * 1/m/(m-1) * sum(sum( (K+L - KL - KL').^2 ));
	 * in MATLAB, so sum up all elements */

	// TODO parallelise or use linalg and precomputed kernel matrix
	float64_t var_mmd=0;
	for (index_t i=0; i<m; ++i)
	{
		for (index_t j=0; j<m; ++j)
		{
			/* dont add diagonal of all pairs of imaginary kernel matrices */
			if (i==j || m+i==j || m+j==i)
				continue;

			float64_t to_add=kernel_matrix(i, j);
			to_add+=kernel_matrix(m+i, m+j);
			to_add-=kernel_matrix(i, m+j);
			to_add-=kernel_matrix(m+i, j);
			var_mmd+=CMath::pow(to_add, 2);
		}
	}

	var_mmd*=2.0/m/(m-1)*1.0/m/(m-1);

	/* parameters for gamma distribution */
	float64_t a=CMath::pow(mean_mmd, 2)/var_mmd;
	float64_t b=var_mmd*m/mean_mmd;

	result[0]=a;
	result[1]=b;

	SG_DEBUG("Leaving\n");
	return result;
}

SGVector<float64_t> CQuadraticTimeMMD::spectrum_sample_null()
{
	SG_DEBUG("Entering\n");
	DataManager& dm=get_data_manager();
	index_t m=dm.num_samples_at(0);
	index_t n=dm.num_samples_at(1);

	if (self->num_eigenvalues > m+n - 1)
	{
		SG_ERROR("Number of Eigenvalues (%d) for spectrum approximation"
				" must be smaller than %d\n", self->num_eigenvalues,
				m+n - 1);
	}

	if (self->num_eigenvalues<1)
	{
		SG_ERROR("Number of Eigenvalues (%d) must be positive.\n",
				self->num_eigenvalues);
	}

	SGVector<float64_t> null_samples(get_num_null_samples());
	std::fill(null_samples.vector, null_samples.vector+null_samples.vlen, 0);

	/* imaginary matrix K=[K KL; KL' L] (MATLAB notation)
	 * K is matrix for XX, L is matrix for YY, KL is XY, LK is YX
	 * works since X and Y are concatenated here */
	SGMatrix<float32_t> precomputed_km=self->get_kernel_matrix();
	SGMatrix<float32_t> K(precomputed_km.num_rows, precomputed_km.num_cols);
	std::copy(precomputed_km.matrix, precomputed_km.matrix+precomputed_km.num_rows*precomputed_km.num_cols, K.matrix);

	/* center matrix K=H*K*H */
	K.center();

	/* compute eigenvalues and select num_eigenvalues largest ones */
	Eigen::Map<Eigen::MatrixXf> c_kernel_matrix(K.matrix, K.num_rows, K.num_cols);
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eigen_solver(c_kernel_matrix);
	REQUIRE(eigen_solver.info()==Eigen::Success, "Eigendecomposition failed!\n");
	index_t max_num_eigenvalues=eigen_solver.eigenvalues().rows();

	/* finally, sample from null distribution */
	for (auto i=0; i<null_samples.vlen; ++i)
	{
		float64_t null_sample=0;
		for (index_t j=0; j<self->num_eigenvalues; ++j)
		{
			float64_t z_j=CMath::randn_double();
			float64_t multiple=CMath::sq(z_j);

			/* take largest EV, scale by 1/(m+n) on the fly and take abs value*/
			float64_t eigenvalue_estimate=eigen_solver.eigenvalues()[max_num_eigenvalues-1-j];
			eigenvalue_estimate/=(m+n);

			if (get_statistic_type()==ST_UNBIASED_FULL)
				multiple-=1;

			null_sample+=eigenvalue_estimate*multiple;
		}
		null_samples[i]=null_sample;
	}

	SG_DEBUG("Leaving\n");
	return null_samples;
}

std::shared_ptr<CCustomDistance> CQuadraticTimeMMD::compute_distance()
{
	auto distance=std::shared_ptr<CCustomDistance>(new CCustomDistance());
	DataManager& dm=get_data_manager();

	// using data manager next() API in order to make it work with
	// streaming samples as well.
	dm.start();
	auto samples=dm.next();
	if (!samples.empty())
	{
		dm.end();

		// use 0th block from each distribution (since there is only one block
		// for quadratic time MMD
		CFeatures *samples_p=samples[0][0].get();
		CFeatures *samples_q=samples[1][0].get();

		try
		{
			auto p_and_q=FeaturesUtil::create_merged_copy(samples_p, samples_q);
			samples.clear();
			auto euclidean_distance=std::unique_ptr<CEuclideanDistance>(new CEuclideanDistance());
			if (euclidean_distance->init(p_and_q, p_and_q))
			{
				auto dist_mat=euclidean_distance->get_distance_matrix<float32_t>();
				if (io->get_loglevel()==MSG_DEBUG)
				{
					dist_mat.display_matrix("distance_matrix");
				}
				distance->set_triangle_distance_matrix_from_full(dist_mat.data(), dist_mat.num_rows, dist_mat.num_cols);
			}
			else
			{
				SG_SERROR("Computing distance matrix was not possible! Please contact Shogun developers.\n");
			}
		}
		catch (ShogunException e)
		{
			SG_SERROR("%s, Data is too large! Computing distance matrix was not possible!\n", e.get_exception_string());
		}
	}
	else
	{
		dm.end();
		SG_SERROR("Could not fetch samples!\n");
	}
	return distance;
}

const char* CQuadraticTimeMMD::get_name() const
{
	return "QuadraticTimeMMD";
}
