/*
* Copyright (c) The Shogun Machine Learning Toolbox
* Written (w) 2016 Fredrik Hallgren
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
#include <shogun/labels/RegressionLabels.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/regression/KRRNystrom.h>
#include <shogun/regression/KernelRidgeRegression.h>
#include <shogun/base/some.h>
#include <gtest/gtest.h>

#ifdef HAVE_CXX11

using namespace shogun;

/**
 * Test the main algorithm by comparison of the alphas and the predictions,
 * using all columns in the approximation to make sure results are very similar
 */
TEST(KRRNystrom, apply_and_compare_to_KRR_with_all_columns)
{
	/* data matrix dimensions */
	index_t num_vectors=10;
	index_t num_features=1;

	/* training label data */
	SGVector<float64_t> lab(num_vectors);

	/* fill data matrix and labels */
	SGMatrix<float64_t> train_dat(num_features, num_vectors);
	SGMatrix<float64_t> test_dat(num_features, num_vectors);
	for (index_t i=0; i<num_vectors; ++i)
	{
		/* labels are linear plus noise */
		lab.vector[i]=i+CMath::normal_random(0, 1.0);
		train_dat.matrix[i]=i;
		test_dat.matrix[i]=i;
	}

	/* training features */
	auto features=some<CDenseFeatures<float64_t>>(train_dat);
	auto features_krr=some<CDenseFeatures<float64_t>>(train_dat);

	/* testing features */
	auto test_features=some<CDenseFeatures<float64_t>>(test_dat);

	/* training labels */
	auto labels=some<CRegressionLabels>(lab);
	auto labels_krr=some<CRegressionLabels>(lab);

	/* kernel */
	auto kernel=some<CGaussianKernel>(features, features, 10, 0.5);
	auto kernel_krr=some<CGaussianKernel>(features_krr, features_krr, 10, 0.5);

	/* kernel ridge regression and the nystrom approximation */
	float64_t tau=0.01;
	auto nystrom=some<CKRRNystrom>(tau, num_vectors, kernel, labels);
	auto krr=some<CKernelRidgeRegression>(tau, kernel_krr, labels_krr);

	nystrom->train();
	krr->train();

	SGVector<float64_t> alphas=nystrom->get_alphas();
	SGVector<float64_t> alphas_krr=krr->get_alphas();

	for (index_t i=0; i<num_vectors; ++i)
		EXPECT_NEAR(alphas[i], alphas_krr[i], 1E-1);

	auto result=nystrom->apply_regression(test_features);
	auto result_krr=krr->apply_regression(test_features);

	for (index_t i=0; i<num_vectors; ++i)
		EXPECT_NEAR(result->get_label(i), result_krr->get_label(i), 1E-5);
}

/**
 * Test the main algorithm by comparison of the alphas and the predictions,
 * using a subset of the columns.
 */
TEST(KRRNystrom, apply_and_compare_to_KRR_with_column_subset)
{
	/* data matrix dimensions */
	index_t num_vectors=100;
	index_t num_features=1;
	index_t num_basis_rkhs=50;

	/* training label data */
	SGVector<float64_t> lab(num_vectors);

	/* fill data matrix and labels */
	SGMatrix<float64_t> train_dat(num_features, num_vectors);
	SGMatrix<float64_t> test_dat(num_features, num_vectors);
	for (index_t i=0; i<num_vectors; ++i)
	{
		/* labels are linear plus noise */
		float64_t point=(float64_t)i*10/num_vectors;
		lab.vector[i]=point+CMath::normal_random(0, 1.0);
		train_dat.matrix[i]=point;
		test_dat.matrix[i]=point;
	}

	/* training features */
	auto features=some<CDenseFeatures<float64_t>>(train_dat);
	auto features_krr=some<CDenseFeatures<float64_t>>(train_dat);

	/* testing features */
	auto test_features=some<CDenseFeatures<float64_t>>(test_dat);

	/* training labels */
	auto labels=some<CRegressionLabels>(lab);
	auto labels_krr=some<CRegressionLabels>(lab);

	/* kernel */
	auto kernel=some<CGaussianKernel>(features, features, 10, 0.5);
	auto kernel_krr=some<CGaussianKernel>(features_krr, features_krr, 10, 0.5);

	/* kernel ridge regression and the nystrom approximation */
	float64_t tau=0.01;
	auto nystrom=some<CKRRNystrom>(tau, num_basis_rkhs, kernel, labels);
	auto krr=some<CKernelRidgeRegression>(tau, kernel_krr, labels_krr);

	nystrom->train();
	krr->train();

	auto result=nystrom->apply_regression(test_features);
	auto result_krr=krr->apply_regression(test_features);

	for (index_t i=0; i<num_vectors; ++i)
		EXPECT_NEAR(result->get_label(i), result_krr->get_label(i), 1E-1);
}

#endif /* HAVE_CXX11 */
