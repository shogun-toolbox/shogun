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

#include <shogun/io/SGIO.h>
#include <shogun/lib/SGVector.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/kernel/CustomKernel.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/statistical_testing/TestEnums.h>
#include <shogun/statistical_testing/QuadraticTimeMMD.h>
#include <shogun/statistical_testing/MultiKernelQuadraticTimeMMD.h>
#include <shogun/statistical_testing/internals/Kernel.h>
#include <shogun/statistical_testing/internals/FeaturesUtil.h>
#include <shogun/statistical_testing/internals/NextSamples.h>
#include <shogun/statistical_testing/internals/DataManager.h>
#include <shogun/statistical_testing/internals/KernelManager.h>
#include <shogun/statistical_testing/internals/mmd/ComputeMMD.h>
#include <shogun/statistical_testing/internals/mmd/PermutationMMD.h>
#include <shogun/statistical_testing/internals/mmd/VarianceH0.h>
#include <shogun/statistical_testing/internals/mmd/VarianceH1.h>

using namespace shogun;
using namespace internal;
using namespace mmd;
using std::unique_ptr;

struct CQuadraticTimeMMD::Self
{
	Self(CQuadraticTimeMMD&);

	void init_statistic_job();
	void init_permutation_job();
	void init_variance_h1_job();
	void init_kernel();
	SGMatrix<float32_t> get_kernel_matrix();

	SGVector<float64_t> sample_null_spectrum();
	SGVector<float64_t> sample_null_permutation();
	SGVector<float64_t> gamma_fit_null();

	CQuadraticTimeMMD& owner;
	unique_ptr<CMultiKernelQuadraticTimeMMD> multi_kernel;

	/**
	 * Whether to precompute the kernel matrix. by default this is true.
	 * It can be changed by the precompute_kernel_matrix() call. Keep in mind that
	 * precompute is always true as long as the underlying kernel itself is a
	 * precomputed kernel. Further updation of this value is ignored unless the
	 * kernel is changed to a non-precomputed one.
	 */
	bool precompute;

	/**
	 * Whether the kernel is initialized with the joint features. If a kernel is
	 * initialized once, then it becomes true. It can then becomes false only when
	 * (a) the features are updated, or
	 * (b) the kernel is updated later, or
	 * (c) the internally precomputed kernel is removed and the underlying kernel is in use.
	 * However, for (a), if the underlying kernel itself is a pre-computed one, it
	 * stays true even when the features are updated. Also, for (b), if the newly
	 * updated kernel is a pre-computed one, then also it stays true.
	 */
	bool is_kernel_initialized;

	index_t num_eigenvalues;

	ComputeMMD statistic_job;
	VarianceH0 variance_h0_job;
	VarianceH1 variance_h1_job;
	PermutationMMD permutation_job;

	static constexpr bool DEFAULT_PRECOMPUTE = true;
	static constexpr index_t DEFAULT_NUM_EIGENVALUES = 10;
};

CQuadraticTimeMMD::Self::Self(CQuadraticTimeMMD& mmd) : owner(mmd)
{
	is_kernel_initialized=false;
	precompute=DEFAULT_PRECOMPUTE;
	num_eigenvalues=DEFAULT_NUM_EIGENVALUES;
}

void CQuadraticTimeMMD::Self::init_statistic_job()
{
	REQUIRE(owner.get_num_samples_p()>0,
		"Number of samples from P (was %s) has to be > 0!\n", owner.get_num_samples_p());
	REQUIRE(owner.get_num_samples_q()>0,
		"Number of samples from Q (was %s) has to be > 0!\n", owner.get_num_samples_q());

	statistic_job.m_n_x=owner.get_num_samples_p();
	statistic_job.m_n_y=owner.get_num_samples_q();
	statistic_job.m_stype=owner.get_statistic_type();
}

void CQuadraticTimeMMD::Self::init_variance_h1_job()
{
	REQUIRE(owner.get_num_samples_p()>0,
		"Number of samples from P (was %s) has to be > 0!\n", owner.get_num_samples_p());
	REQUIRE(owner.get_num_samples_q()>0,
		"Number of samples from Q (was %s) has to be > 0!\n", owner.get_num_samples_q());

	variance_h1_job.m_n_x=owner.get_num_samples_p();
	variance_h1_job.m_n_y=owner.get_num_samples_q();
}

void CQuadraticTimeMMD::Self::init_permutation_job()
{
	REQUIRE(owner.get_num_samples_p()>0,
		"Number of samples from P (was %s) has to be > 0!\n", owner.get_num_samples_p());
	REQUIRE(owner.get_num_samples_q()>0,
		"Number of samples from Q (was %s) has to be > 0!\n", owner.get_num_samples_q());
	REQUIRE(owner.get_num_null_samples()>0,
		"Number of null samples (was %d) has to be > 0!\n", owner.get_num_null_samples());

	permutation_job.m_n_x=owner.get_num_samples_p();
	permutation_job.m_n_y=owner.get_num_samples_q();
	permutation_job.m_stype=owner.get_statistic_type();
	permutation_job.m_num_null_samples=owner.get_num_null_samples();
}

void CQuadraticTimeMMD::Self::init_kernel()
{
	ASSERT(owner.get_kernel());
	if (!is_kernel_initialized)
	{
		ASSERT(owner.get_kernel()->get_kernel_type()!=K_CUSTOM);
		auto samples_p_and_q=owner.get_p_and_q();

		auto kernel=owner.get_kernel();
		kernel->init(samples_p_and_q, samples_p_and_q);
		is_kernel_initialized=true;
		SG_SINFO("Kernel is initialized with joint features of %d total samples!\n", samples_p_and_q->get_num_vectors());
	}
}

SGMatrix<float32_t> CQuadraticTimeMMD::Self::get_kernel_matrix()
{
	ASSERT(precompute);
	ASSERT(owner.get_kernel());
	ASSERT(is_kernel_initialized);

	if (owner.get_kernel()->get_kernel_type()!=K_CUSTOM)
	{
		auto kernel=owner.get_kernel();
		owner.get_kernel_mgr().precompute_kernel_at(0);
		kernel->remove_lhs_and_rhs();
	}

	ASSERT(owner.get_kernel()->get_kernel_type()==K_CUSTOM);
	auto precomputed_kernel=static_cast<CCustomKernel*>(owner.get_kernel());
	return precomputed_kernel->get_float32_kernel_matrix();
}

CQuadraticTimeMMD::CQuadraticTimeMMD() : CMMD()
{
	init();
}

CQuadraticTimeMMD::CQuadraticTimeMMD(CFeatures* samples_from_p, CFeatures* samples_from_q) : CMMD(samples_from_p, samples_from_q)
{
	init();
}

void CQuadraticTimeMMD::init()
{
	self=unique_ptr<Self>(new Self(*this));
	self->multi_kernel=unique_ptr<CMultiKernelQuadraticTimeMMD>(new CMultiKernelQuadraticTimeMMD(this));
}

CQuadraticTimeMMD::~CQuadraticTimeMMD()
{
	CMMD::cleanup();
}

void CQuadraticTimeMMD::set_p(CFeatures* samples_from_p)
{
	if (samples_from_p!=get_p())
	{
		CTwoDistributionTest::set_p(samples_from_p);
		get_kernel_mgr().restore_kernel_at(0);
		self->is_kernel_initialized=false;
		self->multi_kernel->invalidate_precomputed_distance();

		if (get_kernel() && get_kernel()->get_kernel_type()==K_CUSTOM)
		{
			SG_WARNING("Existing kernel is already precomputed. Features provided will be\
					ignored unless the kernel is updated with a non-precomputed one!\n");
			self->is_kernel_initialized=true;
		}
	}
	else
	{
		SG_INFO("Provided features are the same as the existing one. Ignoring!\n");
	}
}

void CQuadraticTimeMMD::set_q(CFeatures* samples_from_q)
{
	if (samples_from_q!=get_q())
	{
		CTwoDistributionTest::set_q(samples_from_q);
		get_kernel_mgr().restore_kernel_at(0);
		self->is_kernel_initialized=false;
		self->multi_kernel->invalidate_precomputed_distance();

		if (get_kernel() && get_kernel()->get_kernel_type()==K_CUSTOM)
		{
			SG_WARNING("Existing kernel is already precomputed. Features provided will be\
					ignored unless the kernel is updated with a non-precomputed one!\n");
			self->is_kernel_initialized=true;
		}
	}
	else
	{
		SG_INFO("Provided features are the same as the existing one. Ignoring!\n");
	}
}

CFeatures* CQuadraticTimeMMD::get_p_and_q()
{
	CFeatures* samples_p_and_q=nullptr;
	REQUIRE(get_p(), "Samples from P are not set!\n");
	REQUIRE(get_q(), "Samples from Q are not set!\n");

	DataManager& data_mgr=get_data_mgr();
	data_mgr.start();
	auto samples=data_mgr.next();
	if (!samples.empty())
	{
		CFeatures *samples_p=samples[0][0].get();
		CFeatures *samples_q=samples[1][0].get();
		samples_p_and_q=FeaturesUtil::create_merged_copy(samples_p, samples_q);
		samples.clear();
	}
	else
	{
		SG_SERROR("Could not fetch samples!\n");
	}
	data_mgr.end();
	return samples_p_and_q;
}

void CQuadraticTimeMMD::set_kernel(CKernel* kernel)
{
	if (kernel!=get_kernel())
	{
		// removing any pre-computed kernel is done in the base already
		CTwoSampleTest::set_kernel(kernel);
		self->is_kernel_initialized=false;

		if (kernel->get_kernel_type()==K_CUSTOM)
		{
			SG_INFO("Setting a precomputed kernel. Features provided will be ignored!\n");
			self->is_kernel_initialized=true;
		}
	}
	else
	{
		SG_INFO("Provided kernel is the same as the existing one. Ignoring!\n");
	}
}

void CQuadraticTimeMMD::select_kernel()
{
	CMMD::select_kernel();
	self->is_kernel_initialized=false;

	ASSERT(get_kernel());
	if (get_kernel()->get_kernel_type()==K_CUSTOM)
	{
		SG_WARNING("Selected kernel is already precomputed. Features provided will be\
				ignored unless the kernel is updated with a non-precomputed one!\n");
		self->is_kernel_initialized=true;
	}
}

float64_t CQuadraticTimeMMD::normalize_statistic(float64_t statistic) const
{
	const index_t Nx=get_num_samples_p();
	const index_t Ny=get_num_samples_q();
	return Nx*Ny*statistic/(Nx+Ny);
}


float64_t CQuadraticTimeMMD::compute_statistic()
{
	SG_DEBUG("Entering\n");
	REQUIRE(get_kernel(), "Kernel is not set!\n");

	self->init_statistic_job();
	self->init_kernel();

	float64_t statistic=0;
	if (self->precompute)
	{
		SGMatrix<float32_t> kernel_matrix=self->get_kernel_matrix();
		statistic=self->statistic_job(kernel_matrix);
	}
	else
	{
		auto kernel=get_kernel();
		if (kernel->get_kernel_type()==K_CUSTOM)
			SG_INFO("Precompute is turned off, but provided kernel is already precomputed!\n");
		auto kernel_functor=internal::Kernel(kernel);
		statistic=self->statistic_job(kernel_functor);
	}

	statistic=normalize_statistic(statistic);

	SG_DEBUG("Leaving\n");
	return statistic;
}

SGVector<float64_t> CQuadraticTimeMMD::Self::sample_null_permutation()
{
	SG_SDEBUG("Entering\n");
	REQUIRE(owner.get_kernel(), "Kernel is not set!\n");

	init_permutation_job();
	init_kernel();

	SGVector<float32_t> result;
	if (precompute)
	{
		SGMatrix<float32_t> kernel_matrix=get_kernel_matrix();
		result=permutation_job(kernel_matrix);
	}
	else
	{
		auto kernel=owner.get_kernel();
		if (kernel->get_kernel_type()==K_CUSTOM)
			SG_SINFO("Precompute is turned off, but provided kernel is already precomputed!\n");
		auto kernel_functor=internal::Kernel(kernel);
		result=permutation_job(kernel_functor);
	}

	SGVector<float64_t> null_samples(result.vlen);
	for (auto i=0; i<result.vlen; ++i)
		null_samples[i]=owner.normalize_statistic(result[i]);

	SG_SDEBUG("Leaving\n");
	return null_samples;
}

SGVector<float64_t> CQuadraticTimeMMD::Self::sample_null_spectrum()
{
	SG_SDEBUG("Entering\n");
	REQUIRE(owner.get_kernel(), "Kernel is not set!\n");
	REQUIRE(precompute, "MMD2_SPECTRUM is not possible without precomputing the kernel matrix!\n");

	index_t m=owner.get_num_samples_p();
	index_t n=owner.get_num_samples_q();

	REQUIRE(num_eigenvalues>0 && num_eigenvalues<m+n-1,
		"Number of Eigenvalues (%d) must be in between [1, %d]\n", num_eigenvalues, m+n-1);

	init_kernel();

	/* imaginary matrix K=[K KL; KL' L] (MATLAB notation)
	 * K is matrix for XX, L is matrix for YY, KL is XY, LK is YX
	 * works since X and Y are concatenated here */
	SGMatrix<float32_t> kernel_matrix=get_kernel_matrix();
	SGMatrix<float32_t> K(kernel_matrix.num_rows, kernel_matrix.num_cols);
	std::copy(kernel_matrix.data(), kernel_matrix.data()+kernel_matrix.size(), K.data());

	/* center matrix K=H*K*H */
	K.center();

	/* compute eigenvalues and select num_eigenvalues largest ones */
	Eigen::Map<Eigen::MatrixXf> c_kernel_matrix(K.matrix, K.num_rows, K.num_cols);
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eigen_solver(c_kernel_matrix);
	REQUIRE(eigen_solver.info()==Eigen::Success, "Eigendecomposition failed!\n");
	index_t max_num_eigenvalues=eigen_solver.eigenvalues().rows();

	SGVector<float64_t> null_samples(owner.get_num_null_samples());

	/* finally, sample from null distribution */
	for (auto i=0; i<null_samples.vlen; ++i)
	{
		float64_t null_sample=0;
		for (index_t j=0; j<num_eigenvalues; ++j)
		{
			float64_t z_j=CMath::randn_double();
			float64_t multiple=CMath::sq(z_j);

			/* take largest EV, scale by 1/(m+n) on the fly and take abs value*/
			float64_t eigenvalue_estimate=eigen_solver.eigenvalues()[max_num_eigenvalues-1-j];
			eigenvalue_estimate/=(m+n);

			if (owner.get_statistic_type()==EStatisticType::UNBIASED_FULL)
				multiple-=1;

			null_sample+=eigenvalue_estimate*multiple;
		}
		null_samples[i]=null_sample;
	}

	SG_SDEBUG("Leaving\n");
	return null_samples;
}

SGVector<float64_t> CQuadraticTimeMMD::Self::gamma_fit_null()
{
	SG_SDEBUG("Entering\n");

	REQUIRE(owner.get_kernel(), "Kernel is not set!\n");
	REQUIRE(precompute, "MMD2_GAMMA is not possible without precomputing the kernel matrix!\n");
	REQUIRE(owner.get_statistic_type()==EStatisticType::BIASED_FULL, "Provided statistic has to be BIASED!\n");

	index_t m=owner.get_num_samples_p();
	index_t n=owner.get_num_samples_q();
	REQUIRE(m==n, "Number of samples from p (%d) and q (%d) must be equal.\n", n, m)

	SGVector<float64_t> result(2);
	std::fill(result.vector, result.vector+result.vlen, 0);

	init_kernel();

	/* imaginary matrix K=[K KL; KL' L] (MATLAB notation)
	 * K is matrix for XX, L is matrix for YY, KL is XY, LK is YX
	 * works since X and Y are concatenated here */
	SGMatrix<float32_t> kernel_matrix=get_kernel_matrix();

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

	SG_SDEBUG("Leaving\n");
	return result;
}

float64_t CQuadraticTimeMMD::compute_variance_h0()
{
	REQUIRE(get_kernel(), "Kernel is not set!\n");
	REQUIRE(self->precompute,
		"Computing variance estimate is not possible without precomputing the kernel matrix!\n");

	self->init_kernel();
	SGMatrix<float32_t> kernel_matrix=self->get_kernel_matrix();
	return self->variance_h0_job(kernel_matrix);
}

float64_t CQuadraticTimeMMD::compute_variance_h1()
{
	REQUIRE(get_kernel(), "Kernel is not set!\n");
	REQUIRE(self->precompute,
		"Computing variance estimate is not possible without precomputing the kernel matrix!\n");

	self->init_kernel();
	self->init_variance_h1_job();
	SGMatrix<float32_t> kernel_matrix=self->get_kernel_matrix();
	return self->variance_h1_job(kernel_matrix);
}

float64_t CQuadraticTimeMMD::compute_p_value(float64_t statistic)
{
	REQUIRE(get_kernel(), "Kernel is not set!\n");
	float64_t result=0;
	switch (get_null_approximation_method())
	{
		case ENullApproximationMethod::MMD2_GAMMA:
		{
			SGVector<float64_t> params=self->gamma_fit_null();
			result=CStatistics::gamma_cdf(statistic, params[0], params[1]);
			break;
		}
		default:
			result=CHypothesisTest::compute_p_value(statistic);
		break;
	}
	return result;
}

float64_t CQuadraticTimeMMD::compute_threshold(float64_t alpha)
{
	REQUIRE(get_kernel(), "Kernel is not set!\n");
	float64_t result=0;
	switch (get_null_approximation_method())
	{
		case ENullApproximationMethod::MMD2_GAMMA:
		{
			SGVector<float64_t> params=self->gamma_fit_null();
			result=CStatistics::gamma_inverse_cdf(alpha, params[0], params[1]);
			break;
		}
		default:
			result=CHypothesisTest::compute_threshold(alpha);
			break;
	}
	return result;
}

SGVector<float64_t> CQuadraticTimeMMD::sample_null()
{
	REQUIRE(get_kernel(), "Kernel is not set!\n");
	SGVector<float64_t> null_samples;
	switch (get_null_approximation_method())
	{
		case ENullApproximationMethod::MMD2_SPECTRUM:
			null_samples=self->sample_null_spectrum();
			break;
		case ENullApproximationMethod::PERMUTATION:
			null_samples=self->sample_null_permutation();
			break;
		default: break;
	}
	return null_samples;
}

CMultiKernelQuadraticTimeMMD* CQuadraticTimeMMD::multikernel()
{
	return self->multi_kernel.get();
}

void CQuadraticTimeMMD::spectrum_set_num_eigenvalues(index_t num_eigenvalues)
{
	self->num_eigenvalues=num_eigenvalues;
}

index_t CQuadraticTimeMMD::spectrum_get_num_eigenvalues() const
{
	return self->num_eigenvalues;
}

void CQuadraticTimeMMD::precompute_kernel_matrix(bool precompute)
{
	if (self->precompute && !precompute)
	{
		if (get_kernel())
		{
			get_kernel_mgr().restore_kernel_at(0);
			self->is_kernel_initialized=false;
			if (get_kernel()->get_kernel_type()==K_CUSTOM)
			{
				SG_WARNING("The existing kernel itself is a precomputed kernel!\n");
				precompute=true;
				self->is_kernel_initialized=true;
			}
		}
	}
	self->precompute=precompute;
}

const char* CQuadraticTimeMMD::get_name() const
{
	return "QuadraticTimeMMD";
}
