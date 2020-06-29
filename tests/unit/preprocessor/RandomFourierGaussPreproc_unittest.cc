/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Author: Nanubala Gnana Sai
 */

#include <gtest/gtest.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/preprocessor/RandomFourierGaussPreproc.h>

using namespace shogun;
using namespace random;

class RFGPTest : public ::testing::Test
{
public:
	virtual void SetUp()
	{
		SGMatrix<float64_t> mat(num_features, num_vectors);
		linalg::range_fill(mat, 1.0);

		auto gauss = std::make_shared<GaussianKernel>(width);
		auto features = std::make_shared<DenseFeatures<float64_t>>(mat);

		preproc = std::make_shared<RandomFourierGaussPreproc>();
		preproc->put(kSeed, seed);
		preproc->set_kernel(gauss);
		preproc->set_dim_output(target_dim);
		preproc->fit(features);
	}
	virtual void TearDown()
	{
	}

protected:
	const int32_t seed = 100;
	const index_t num_vectors = 5;
	const index_t num_features = 3;
	const index_t target_dim = 400;
	const float64_t width = 1.5;
	const float64_t epsilon = 0.04;
	std::shared_ptr<RandomFourierGaussPreproc> preproc;
};

TEST_F(RFGPTest, apply)
{
	SGMatrix<float64_t> matrix(num_features, 2);
	linalg::range_fill(matrix, 1.0);
	auto feats = std::make_shared<DenseFeatures<float64_t>>(matrix);
	auto preprocessed = preproc->transform(feats)
	                        ->as<DenseFeatures<float64_t>>()
	                        ->get_feature_matrix();

	auto result_rff =
	    linalg::dot(preprocessed.get_column(0), preprocessed.get_column(1));

	auto gauss_kernel = std::make_shared<GaussianKernel>();
	gauss_kernel->set_width(width);
	gauss_kernel->init(feats, feats);

	auto result_kernel = gauss_kernel->kernel(0, 1);
	EXPECT_NEAR(result_rff, result_kernel, epsilon);
}

TEST_F(RFGPTest, apply_to_vectors)
{
	SGVector<float64_t> vec1 = {1.0, 2.0, 3.0};
	SGVector<float64_t> vec2 = {4.0, 5.0, 6.0};
	auto mat = SGMatrix<float64_t>(num_features, 2);
	linalg::range_fill(mat, 1.0);

	auto processed1 = preproc->apply_to_feature_vector(vec1);
	auto processed2 = preproc->apply_to_feature_vector(vec2);

	auto result_rff = linalg::dot(processed1, processed2);
	auto gauss_kernel = std::make_shared<GaussianKernel>();
	auto feats = std::make_shared<DenseFeatures<float64_t>>(mat);
	gauss_kernel->set_width(width);
	gauss_kernel->init(feats, feats);

	auto result_kernel = gauss_kernel->kernel(0, 1);
	EXPECT_NEAR(result_rff, result_kernel, epsilon);
}
