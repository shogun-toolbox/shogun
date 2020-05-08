/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include <gtest/gtest.h>
#include <shogun/kernel/MaternKernel.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/util/zip_iterator.h>

using namespace shogun;

TEST(MaternKernelTest_1_2, test_kernel_matrix)
{
	constexpr float64_t width = 2.0;
	constexpr float64_t nu = 0.5;
	constexpr int32_t cache_size = 10;

	SGMatrix<float64_t> matrix{{0, 1, 2}, {3, 4, 5}};
	SGMatrix<float64_t> expected_kernel_matrix{{1., 0.07441661},
	                                           {0.07441661, 1.}};

	auto features = std::make_shared<DenseFeatures<float64_t>>(matrix);

	auto kernel = std::make_shared<MaternKernel>(
	    features, features, cache_size, width, nu);
	SGMatrix<float64_t> computed_kernel_matrix = kernel->get_kernel_matrix();

	for (const auto & [ result, reference ] :
	     zip_iterator(computed_kernel_matrix, expected_kernel_matrix))
		EXPECT_NEAR(result, reference, 1E-7);
}

TEST(MaternKernelTest_3_2, test_kernel_matrix)
{
	constexpr float64_t width = 2.0;
	constexpr float64_t nu = 1.5;
	constexpr int32_t cache_size = 10;

	SGMatrix<float64_t> matrix{{0, 1, 2}, {3, 4, 5}};
	SGMatrix<float64_t> expected_kernel_matrix{{1., 0.06109951},
	                                           {0.06109951, 1.}};

	auto features = std::make_shared<DenseFeatures<float64_t>>(matrix);

	auto kernel = std::make_shared<MaternKernel>(
	    features, features, cache_size, width, nu);
	SGMatrix<float64_t> computed_kernel_matrix = kernel->get_kernel_matrix();

	for (const auto & [ result, reference ] :
	     zip_iterator(computed_kernel_matrix, expected_kernel_matrix))
		EXPECT_NEAR(result, reference, 1E-7);
}

TEST(MaternKernelTest_5_2, test_kernel_matrix)
{
	constexpr float64_t width = 2.0;
	constexpr float64_t nu = 2.5;
	constexpr int32_t cache_size = 10;

	SGMatrix<float64_t> matrix{{0, 1, 2}, {3, 4, 5}};
	SGMatrix<float64_t> expected_kernel_matrix{{1., 0.05416044},
	                                           {0.05416044, 1.}};

	auto features = std::make_shared<DenseFeatures<float64_t>>(matrix);

	auto kernel = std::make_shared<MaternKernel>(
	    features, features, cache_size, width, nu);
	SGMatrix<float64_t> computed_kernel_matrix = kernel->get_kernel_matrix();

	for (const auto & [ result, reference ] :
	     zip_iterator(computed_kernel_matrix, expected_kernel_matrix))
		EXPECT_NEAR(result, reference, 1E-7);
}

TEST(MaternKernelTest_nu, test_kernel_matrix)
{
	constexpr float64_t width = 2.0;
	constexpr float64_t nu = 1.0;
	constexpr int32_t cache_size = 10;

	SGMatrix<float64_t> matrix{{0, 1, 2}, {3, 4, 5}};
	SGMatrix<float64_t> expected_kernel_matrix{{1., 0.06673040739011263},
	                                           {0.06673040739011263, 1.}};

	auto features = std::make_shared<DenseFeatures<float64_t>>(matrix);

	auto kernel = std::make_shared<MaternKernel>(
	    features, features, cache_size, width, nu);
	SGMatrix<float64_t> computed_kernel_matrix = kernel->get_kernel_matrix();

	for (const auto & [ result, reference ] :
	     zip_iterator(computed_kernel_matrix, expected_kernel_matrix))
		EXPECT_NEAR(result, reference, 1E-7);
}

TEST(MaternKernelTest_1_2, test_derivative_width)
{
	// gradient computed with JAX:
	// https://gist.github.com/gf712/b10ecefd2379e8492c9802ac8cb329c9
	constexpr float64_t width = 2.0;
	constexpr float64_t nu = 0.5;
	constexpr int32_t cache_size = 10;

	SGMatrix<float64_t> matrix{{0, 1, 2}, {3, 4, 5}};
	SGMatrix<float64_t> expected_dmatrix{{0., 0.09667}, {0.09667, 0.}};

	auto features = std::make_shared<DenseFeatures<float64_t>>(matrix);

	auto kernel = std::make_shared<MaternKernel>(
	    features, features, cache_size, width, nu);
	auto params = kernel->get_params();
	auto width_param = params.find("width");
	SGMatrix<float64_t> dMatrix = kernel->get_parameter_gradient(*width_param);

	for (const auto & [ result, reference ] :
	     zip_iterator(dMatrix, expected_dmatrix))
		EXPECT_NEAR(result, reference, 1E-7);
}

TEST(MaternKernelTest_3_2, test_derivative_width)
{
	constexpr float64_t width = 2.0;
	constexpr float64_t nu = 1.5;
	constexpr int32_t cache_size = 10;

	SGMatrix<float64_t> matrix{{0, 1, 2}, {3, 4, 5}};
	SGMatrix<float64_t> expected_dmatrix{{0., 0.11247862}, {0.11247862, 0.}};

	auto features = std::make_shared<DenseFeatures<float64_t>>(matrix);

	auto kernel = std::make_shared<MaternKernel>(
	    features, features, cache_size, width, nu);
	auto params = kernel->get_params();
	auto width_param = params.find("width");
	SGMatrix<float64_t> dMatrix = kernel->get_parameter_gradient(*width_param);

	for (const auto & [ result, reference ] :
	     zip_iterator(dMatrix, expected_dmatrix))
		EXPECT_NEAR(result, reference, 1E-7);
}

TEST(MaternKernelTest_5_2, test_derivative_width)
{
	constexpr float64_t width = 2.0;
	constexpr float64_t nu = 2.5;
	constexpr int32_t cache_size = 10;

	SGMatrix<float64_t> matrix{{0, 1, 2}, {3, 4, 5}};
	SGMatrix<float64_t> expected_dmatrix{{0., 0.11487176}, {0.11487176, 0.}};

	auto features = std::make_shared<DenseFeatures<float64_t>>(matrix);

	auto kernel = std::make_shared<MaternKernel>(
	    features, features, cache_size, width, nu);
	auto params = kernel->get_params();
	auto width_param = params.find("width");
	SGMatrix<float64_t> dMatrix = kernel->get_parameter_gradient(*width_param);

	for (const auto & [ result, reference ] :
	     zip_iterator(dMatrix, expected_dmatrix))
		EXPECT_NEAR(result, reference, 1E-7);
}

TEST(MaternKernelTest_nu, test_derivative_width)
{
	constexpr float64_t epsilon = 1E-6;
	constexpr float64_t width = 2.0;
	constexpr float64_t nu = 1.0;
	constexpr int32_t cache_size = 10;

	SGMatrix<float64_t> matrix{{0, 1, 2}, {3, 4, 5}};
	auto features = std::make_shared<DenseFeatures<float64_t>>(matrix);

	auto kernel = std::make_shared<MaternKernel>(
	    features, features, cache_size, width, nu);
	auto params = kernel->get_params();
	auto width_param = params.find("width");
	SGMatrix<float64_t> dMatrix = kernel->get_parameter_gradient(*width_param);

	SGMatrix<float64_t> computed_kernel_matrix = kernel->get_kernel_matrix();
	kernel->put("width", width + epsilon);
	SGMatrix<float64_t> computed_kernel_matrix_epsilon =
	    kernel->get_kernel_matrix();

	for (const auto & [ result, reference, reference_epsilon ] : zip_iterator(
	         dMatrix, computed_kernel_matrix, computed_kernel_matrix_epsilon))
		EXPECT_NEAR(result, (reference_epsilon - reference) / epsilon, 1E-7);
}
