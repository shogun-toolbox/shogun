/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2016 Heiko Strathmann
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
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

#include <shogun/features/DenseFeatures.h>
#include <shogun/base/init.h>
#include <shogun/distributions/kernel_exp_family/impl/Full.h>
#include <shogun/distributions/kernel_exp_family/impl/Nystrom.h>
#include <shogun/distributions/kernel_exp_family/impl/NystromD.h>
#include <shogun/distributions/kernel_exp_family/impl/kernel/Gaussian.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/base/some.h>
#include <gtest/gtest.h>
#include <memory>

#include "DataFixture.h"

using namespace std;
using namespace shogun;
using namespace kernel_exp_family_impl;

using namespace shogun;
using namespace Eigen;

/** Nystrom test type, used to instantiate different version of the Nystrom
 * solver, see fixture below.
 */
enum class ENystromTestType
{
	E_EXPLCIT_BASIS,
	E_SUBSAMPLED_BASIS,
	E_D_SUBSAMPLED_BASIS,
	E_D_EXPLCIT_BASIS,
	E_D_EXPLCIT_BASIS_NOT_REDUNDANT,

};

/** To test all Nystrom variants against reference results on equal setup.
 *
 * All unit tests are based on the following gist
 * https://gist.github.com/karlnapf/c0b24fc18d946cc315733ed679e249e8
 */
class NystromFixed: public DataFixture, public ::testing::TestWithParam<ENystromTestType>
{
public:
	void SetUp()
	{
		DataFixture::SetUp();
		auto sigma = 2.0;
		auto lambda = 1.0;

		auto X = get_data_train();
		auto kernel = make_shared<kernel::Gaussian>(sigma);

		// value parametrised test to make sure all Nystrom solvers lead to the
		// same results
		auto test_type = GetParam();
		switch (test_type)
		{
			case ENystromTestType::E_EXPLCIT_BASIS:
			{
				num_basis = 2;
				system_size = num_basis*D;

				// explicit basis, manually sub-sampled
				SGMatrix < float64_t > basis(D, num_basis);
				for (auto i = 0; i < system_size; i++)
					basis.matrix[i] = get_data_train().matrix[i];

				est = make_shared <Nystrom> (X, basis, kernel, lambda);
				break;
			}
			case ENystromTestType::E_SUBSAMPLED_BASIS:
			{
				num_basis = 2;
				system_size = num_basis*D;

				SGVector <index_t> basis(num_basis);
				basis[0]=0;
				basis[1]=1;
				est = make_shared <Nystrom> (X, basis, kernel, lambda);
				break;
			}
			case ENystromTestType::E_D_SUBSAMPLED_BASIS:
			{
				num_basis = 4;
				system_size = num_basis;

				SGMatrix<bool> basis_mask(X.num_rows, X.num_cols);
				basis_mask.zero();
				basis_mask(0,0)=1;
				basis_mask(1,0)=1;
				basis_mask(0,1)=1;
				basis_mask(1,1)=1;
				est = make_shared <NystromD> (X, basis_mask, kernel, lambda);
				break;
			}
			case ENystromTestType::E_D_EXPLCIT_BASIS:
			{
				num_basis = 4;
				system_size = num_basis;

				// explicit basis, being all of the training data
				SGMatrix <float64_t> basis = get_data_train().clone();

				SGMatrix<bool> basis_mask(X.num_rows, X.num_cols);
				basis_mask.zero();
				basis_mask(0,0)=1;
				basis_mask(1,0)=1;
				basis_mask(0,1)=1;
				basis_mask(1,1)=1;
				est = make_shared <NystromD> (X, basis, basis_mask, kernel, lambda);
				break;
			}
			case ENystromTestType::E_D_EXPLCIT_BASIS_NOT_REDUNDANT:
			{
				num_basis = 4;
				system_size = num_basis;

				// explicit basis, only part of the training data
				SGMatrix <float64_t> basis(X.num_rows, 2);
				for (auto i=0; i<D*2; i++)
					basis.matrix[i] = X.matrix[i];

				SGMatrix<bool> basis_mask(basis.num_rows, basis.num_cols);
				basis_mask.zero();
				basis_mask(0,0)=1;
				basis_mask(1,0)=1;
				basis_mask(0,1)=1;
				basis_mask(1,1)=1;
				est = make_shared <NystromD> (X, basis, basis_mask, kernel, lambda);
				break;
			}
		}
	}

	virtual SGMatrix<float64_t> get_data_train()
	{
		return X_train_fixed;
	}

protected:
	shared_ptr<Nystrom> est;
	index_t num_basis;
	index_t system_size;
};


class NystromRandom: public NystromFixed
{
	virtual SGMatrix<float64_t> get_data_train()
	{
		return X_train_random;
	}
};

/** To test sub-sampling components (other tests use all components).
 * This is only testing the cases where not all component of a basis point are
 * used. Tests are self referencing the standard Nystrom. */
class KernelExpFamilyImplNystromD: public DataFixture, public ::testing::Test
{
	void SetUp()
	{
		DataFixture::SetUp();
		auto sigma = 2.0;
		auto lambda = 1.0;

		auto kernel = make_shared<kernel::Gaussian>(sigma);

		// Bernoulli sampling for each basis component
		basis_mask=SGMatrix<bool>(X_train_random.num_rows, X_train_random.num_cols);
		basis_mask.zero();
		system_size=0;
		for (auto i=0; i<ND; i++)
		{
			auto rand = CMath::randn_double();
			if (rand>0)
			{
				basis_mask.matrix[i]=true;
				system_size++;
			}
		}

		est = make_shared<NystromD>(X_train_random, basis_mask, kernel, lambda);
		est->fit();

		// create standard nystrom instance that gives the same results, via
		// sub-sampling the beta and setting un-use components to zero
		auto kernel2 = make_shared<kernel::Gaussian>(sigma);
		est_nystrom_classic = make_shared<Nystrom>(X_train_random,
				X_train_random, kernel2, lambda);

		// extract the relevant coefficients into the standard Nystrom
		est_nystrom_classic->set_beta(SGVector<float64_t>(basis_mask.num_rows*
										basis_mask.num_cols));
		est_nystrom_classic->get_beta().zero();
		auto idx=0;
		for (auto a=0; a<N; a++)
		{
			for (auto i=0; i<D; i++)
			{
				if (basis_mask(i,a))
					est_nystrom_classic->get_beta()[a*D+i]=est->get_beta()[idx++];
			}
		}
	}

protected:
	SGMatrix<bool> basis_mask;
	shared_ptr<NystromD> est;
	shared_ptr<Nystrom> est_nystrom_classic;
	index_t system_size;
};

TEST_P(NystromFixed, compute_G_mm)
{
	auto result = est->compute_G_mm();
	ASSERT_EQ(result.num_rows, system_size);
	ASSERT_EQ(result.num_cols, system_size);
	ASSERT_TRUE(result.data());

	// note: matrix is symmetric
	float64_t reference[] = {
		1. , 0. , -0.0045103175789327,
		-0.0090206351578654, 0. , 1. ,
		-0.0090206351578654, -0.0120275135438206, -0.0045103175789327,
		-0.0090206351578654, 1. , 0. ,
		-0.0090206351578654, -0.0120275135438206, 0. , 1.
	};

	for (auto i=0; i<system_size*system_size; i++)
		EXPECT_NEAR(result[i], reference[i], 1e-15);
}

TEST_P(NystromFixed, compute_G_mn)
{
	auto result = est->compute_G_mn();
	ASSERT_EQ(result.num_rows, system_size);
	ASSERT_EQ(result.num_cols, ND);
	ASSERT_TRUE(result.data());

	float64_t reference[] = {
		1.0000000000000000e+00, 0.0000000000000000e+00,
		-4.5103175789327175e-03, -9.0206351578654351e-03,
		0.0000000000000000e+00, 1.0000000000000000e+00,
		-9.0206351578654351e-03, -1.2027513543820579e-02,
		-4.5103175789327175e-03, -9.0206351578654351e-03,
		1.0000000000000000e+00, 0.0000000000000000e+00,
		-9.0206351578654351e-03, -1.2027513543820579e-02,
		0.0000000000000000e+00, 1.0000000000000000e+00,
		-3.3119501750281335e-07, -6.2099065781777500e-07,
		0.0000000000000000e+00, -1.6416999724779760e-01,
		-6.2099065781777500e-07, -9.9358505250844009e-07,
		-1.6416999724779760e-01,  -2.4625499587169641e-01
	};

	for (auto i=0; i<system_size*system_size; i++)
		EXPECT_NEAR(result[i], reference[i], 1e-15);
}

TEST_P(NystromFixed, compute_system_matrix)
{
	auto result = est->compute_system_matrix();
	ASSERT_EQ(result.num_rows, system_size);
	ASSERT_EQ(result.num_cols, system_size);
	ASSERT_TRUE(result.data());

	// note: matrix is symmetric
	float64_t reference[] = {
		1.3333672382746031e+00, 4.9727247227808545e-05,
		-7.5171619822096674e-03, -1.5034322831663395e-02,
		4.9727247227808545e-05, 1.3334086776473568e+00,
		-1.5034337557490613e-02, -2.0045740365261768e-02,
		-7.5171619822096674e-03, -1.5034337557490613e-02,
		1.3423511676065520e+00, 1.3525621245124521e-02,
		-1.5034322831663395e-02, -2.0045740365261768e-02,
		1.3525621245124521e-02, 1.3626064479762696e+00
	};

	for (auto i=0; i<result.num_rows*result.num_cols; i++)
		EXPECT_NEAR(result[i], reference[i], 1e-15);
}

TEST_P(NystromFixed, compute_system_vector)
{
	auto result = est->compute_system_vector();
	ASSERT_EQ(result.vlen, system_size);
	ASSERT_TRUE(result.vector);

	float64_t reference[] = {
		0.0090218771391811, 0.0135330227056575, 0.0183410310501008,
		0.0411923796791344
	};

	for (auto i=0; i<system_size; i++)
		EXPECT_NEAR(result[i], reference[i], 1e-15);
}

TEST_P(NystromFixed, fit_kernel_Gaussian)
{
	est->fit();
	auto result = est->get_beta();
	ASSERT_EQ(result.vlen, system_size);
	ASSERT_TRUE(result.vector);

	float64_t reference[] = {
		-0.0071840764907642, -0.010757370959334, -0.0135184296925311,
		-0.0303339102579069
	};

	for (auto i=0; i<system_size; i++)
		EXPECT_NEAR(result[i], reference[i], 1e-15);

}

TEST_P(NystromFixed, log_pdf_kernel_Gaussian)
{
	est->fit();
	est->set_data(X_test_fixed);
	auto log_pdf = est->log_pdf();

	ASSERT_TRUE(log_pdf.vector != NULL);
	ASSERT_EQ(log_pdf.vlen, N_test);
	EXPECT_NEAR(log_pdf[0], 0.0001774638427285, 1e-15);
	EXPECT_NEAR(log_pdf[1], -0.0036531113518117, 1e-15);
}

TEST_P(NystromFixed, grad_kernel_Gaussian)
{
	est->fit();
	est->set_data(X_test_fixed);
	auto grad = est->grad(0);

	float64_t reference[] = {-0.0068494729423344, -0.0102705846207064};
	ASSERT_EQ(grad.vlen, D);
	ASSERT_TRUE(grad.vector);
	for (auto i=0; i<D; i++)
		EXPECT_NEAR(grad[i], reference[i], 1e-15);

	grad = est->grad(1);
	float64_t reference2[] = {0.0006131648387784, -0.0046163096796586};
	ASSERT_EQ(grad.vlen, D);
	ASSERT_TRUE(grad.vector);
	for (auto i=0; i<D; i++)
		EXPECT_NEAR(grad[i], reference2[i], 1e-15);
}

TEST_P(NystromFixed, hessian_kernel_Gaussian)
{
	est->fit();
	est->set_data(X_test_fixed);
	auto hessian = est->hessian(0);

	float64_t reference[] = {
		0.0004510949800765, 0.0009126002661734,
		0.0009126002661734, 0.0011460796044802
	};
	for (auto i=0; i<D*D; i++)
		EXPECT_NEAR(hessian[i], reference[i], 1e-8);

	hessian = est->hessian(1);

	float64_t reference2[] = {
		0.0085325523811802, 0.0081597815414807,
		0.0081597815414807, 0.0087650433882726
	};

	for (auto i=0; i<D*D; i++)
		EXPECT_NEAR(hessian[i], reference2[i], 1e-8);
}

TEST_P(NystromRandom, hessian_diag_equals_hessian)
{
	est->fit();
	est->set_data(X_test_fixed);
	for (auto i=0; i<est->get_num_data(); i++)
	{
		auto hessian = est->hessian(i);
		auto hessian_diag = est->hessian_diag(i);

		ASSERT_EQ(hessian.num_rows, D);
		ASSERT_EQ(hessian.num_cols, D);
		ASSERT_TRUE(hessian.data());

		ASSERT_EQ(hessian_diag.vlen, D);
		ASSERT_TRUE(hessian_diag.data());

		for (auto j=0; j<D; j++)
		EXPECT_NEAR(hessian_diag[j], hessian(j,j), 1e-8);
	}
}

TEST_P(NystromFixed, score_kernel_Gaussian)
{
	est->fit();
	EXPECT_NEAR(est->score(), -0.0014814034043, 1e-14);

	est->set_data(X_test_fixed);
	EXPECT_NEAR(est->score(), 0.00949090679556, 1e-14);
}

INSTANTIATE_TEST_CASE_P(KernelExpFamilyImpl,
						NystromFixed,
						::testing::Values(
								ENystromTestType::E_EXPLCIT_BASIS,
								ENystromTestType::E_SUBSAMPLED_BASIS,
								ENystromTestType::E_D_SUBSAMPLED_BASIS,
								ENystromTestType::E_D_EXPLCIT_BASIS,
								ENystromTestType::E_D_EXPLCIT_BASIS_NOT_REDUNDANT)
);

INSTANTIATE_TEST_CASE_P(KernelExpFamilyImpl,
						NystromRandom,
						::testing::Values(
								ENystromTestType::E_EXPLCIT_BASIS,
								ENystromTestType::E_SUBSAMPLED_BASIS,
								ENystromTestType::E_D_SUBSAMPLED_BASIS,
								ENystromTestType::E_D_EXPLCIT_BASIS,
								ENystromTestType::E_D_EXPLCIT_BASIS_NOT_REDUNDANT)
);

TEST_F(KernelExpFamilyImplNystromD, fit)
{
}

TEST_F(KernelExpFamilyImplNystromD, log_pdf)
{
	for (auto a=0; a<N; a++)
		EXPECT_NEAR(est->log_pdf(a), est_nystrom_classic->log_pdf(a), 1e-15);
}

TEST_F(KernelExpFamilyImplNystromD, grad)
{
	for (auto a=0; a<N; a++)
	{
		auto grad = est->grad(a);
		auto grad2 = est_nystrom_classic->grad(a);
		ASSERT_EQ(grad.vlen, D);
		ASSERT_EQ(grad2.vlen, D);
		ASSERT_TRUE(grad.data());
		ASSERT_TRUE(grad2.data());

		for (auto i=0; i<D; i++)
			EXPECT_NEAR(grad[i], grad2[i], 1e-15);
	}
}

TEST_F(KernelExpFamilyImplNystromD, hessian)
{
	for (auto a=0; a<N; a++)
	{
		auto hessian = est->hessian(a);
		auto hessian2 = est_nystrom_classic->hessian(a);
		ASSERT_EQ(hessian.num_rows, D);
		ASSERT_EQ(hessian.num_cols, D);
		ASSERT_EQ(hessian2.num_rows, D);
		ASSERT_EQ(hessian2.num_cols, D);
		ASSERT_TRUE(hessian.data());
		ASSERT_TRUE(hessian2.data());

		for (auto i=0; i<D*D; i++)
			EXPECT_NEAR(hessian.matrix[i], hessian2.matrix[i], 1e-15);
	}
}

TEST_F(KernelExpFamilyImplNystromD, hessian_diag)
{
	for (auto a=0; a<N; a++)
	{
		auto hessian = est->hessian_diag(a);
		auto hessian2 = est_nystrom_classic->hessian_diag(a);
		ASSERT_EQ(hessian.vlen, D);
		ASSERT_EQ(hessian2.vlen, D);
		ASSERT_TRUE(hessian.data());
		ASSERT_TRUE(hessian2.data());

		for (auto i=0; i<D; i++)
			EXPECT_NEAR(hessian[i], hessian2[i], 1e-15);
	}
}

TEST_F(KernelExpFamilyImplNystromD, score)
{
	EXPECT_NEAR(est->score(), est_nystrom_classic->score(), 1e-15);
}

TEST_F(KernelExpFamilyImplNystromD, compute_h)
{
	auto h = est->compute_h();
	auto h2 = est_nystrom_classic->compute_h();

	ASSERT_EQ(h.vlen, system_size);
	ASSERT_TRUE(h.data());

	ASSERT_EQ(h2.vlen, ND);
	ASSERT_TRUE(h2.data());

	auto basis_inds = est->basis_inds_from_mask(basis_mask);

	// check that vector is subsampled version of the std nystrom one
	for (auto idx=0; idx<system_size; idx++)
		EXPECT_NEAR(h[idx], h2[basis_inds[idx]], 1e-15);
}

TEST_F(KernelExpFamilyImplNystromD, compute_G_mn)
{
	auto G_mn = est->compute_G_mn();
	auto G_mn2 = est_nystrom_classic->compute_G_mn();

	ASSERT_EQ(G_mn.num_rows, system_size);
	ASSERT_EQ(G_mn.num_cols, ND);
	ASSERT_TRUE(G_mn.data());

	ASSERT_EQ(G_mn2.num_rows, ND);
	ASSERT_EQ(G_mn2.num_cols, ND);
	ASSERT_TRUE(G_mn2.data());

	auto basis_inds = est->basis_inds_from_mask(basis_mask);

	// check that matrix is subsampled version of the std nystrom one
	for (auto idx=0; idx<system_size; idx++)
	{
		for (auto idx_full=0; idx_full<ND; idx_full++)
			EXPECT_NEAR(G_mn(idx, idx_full), G_mn2(basis_inds[idx], idx_full), 1e-15);
	}
}

TEST_F(KernelExpFamilyImplNystromD, compute_G_mm)
{
	auto G_mm = est->compute_G_mm();
	auto G_mm2 = est_nystrom_classic->compute_G_mm();

	ASSERT_EQ(G_mm.num_rows, system_size);
	ASSERT_EQ(G_mm.num_cols, system_size);
	ASSERT_TRUE(G_mm.data());

	ASSERT_EQ(G_mm2.num_rows, ND);
	ASSERT_EQ(G_mm2.num_cols, ND);
	ASSERT_TRUE(G_mm2.data());

	auto basis_inds = est->basis_inds_from_mask(basis_mask);

	// check that matrix is subsampled version of the std nystrom one
	for (auto idx_k=0; idx_k<system_size; idx_k++)
	{
		for (auto idx_l=0; idx_l<system_size; idx_l++)
			EXPECT_NEAR(G_mm(idx_k, idx_l), G_mm2(basis_inds[idx_k], basis_inds[idx_l]), 1e-15);
	}
}

TEST(KernelExpFamilyImplNystromDStatic, idx_to_ai)
{
	index_t D=3;

	index_t idx=0;
	auto ai=NystromD::idx_to_ai(idx, D);
	EXPECT_EQ(ai.first, 0);
	EXPECT_EQ(ai.second, 0);

	idx=1;
	ai=NystromD::idx_to_ai(idx, D);
	EXPECT_EQ(ai.first, 0);
	EXPECT_EQ(ai.second, 1);

	idx=2;
	ai=NystromD::idx_to_ai(idx, D);
	EXPECT_EQ(ai.first, 0);
	EXPECT_EQ(ai.second, 2);

	idx=3;
	ai=NystromD::idx_to_ai(idx, D);
	EXPECT_EQ(ai.first, 1);
	EXPECT_EQ(ai.second, 0);

	idx=4;
	ai=NystromD::idx_to_ai(idx, D);
	EXPECT_EQ(ai.first, 1);
	EXPECT_EQ(ai.second, 1);
}

TEST(KernelExpFamilyImplNystromStatic, pinv_self_adjoint)
{
	// TODO this should go to linalg
	index_t N=3;
	index_t D=2;
	SGMatrix<float64_t> X(D,N);
	X(0,0)=0;
	X(1,0)=1;
	X(0,1)=2;
	X(1,1)=4;
	X(0,2)=3;
	X(1,2)=1;

	SGMatrix<float64_t> S(D, D);

	auto eigen_X = Map<MatrixXd>(X.matrix, D, N);
	auto eigen_S = Map<MatrixXd>(S.matrix, D, D);
	eigen_S = eigen_X*eigen_X.adjoint();

	auto pinv = Nystrom::pinv_self_adjoint(S);

	ASSERT_EQ(pinv.num_rows, 2);
	ASSERT_EQ(pinv.num_cols, 2);

	// from numpy.linalg.pinv
	float64_t reference[] = {0.15929204, -0.09734513, -0.09734513, 0.11504425};

	for (auto i=0; i<pinv.num_rows*pinv.num_cols; i++)
	EXPECT_NEAR(pinv[i], reference[i], 1e-8);
}

