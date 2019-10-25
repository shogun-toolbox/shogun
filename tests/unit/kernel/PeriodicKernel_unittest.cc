/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Esben Sorig, Heiko Strathmann
 */
#include <gtest/gtest.h>
#include <shogun/kernel/PeriodicKernel.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/features/DenseFeatures.h>

using namespace shogun;

static void is_eqauls(const SGMatrix<float64_t> a, SGMatrix<float64_t> b, float64_t tolerance = 1E-15)
{
	EXPECT_TRUE((a.num_rows == b.num_rows) && (a.num_cols == b.num_cols));

	for (int64_t i=0; i<int64_t(a.num_rows)*a.num_cols; ++i)
		EXPECT_NEAR(a.matrix[i], b.matrix[i], tolerance);
}

TEST(PeriodicKernelTest,test_kernel_matrix)
{
	// Construct matrix with features
	SGMatrix<float64_t> matrix(2,3);
	for (int32_t i=0; i<6; i++) {
		matrix.matrix[i]=i;
	}

	// Load them into DenseFeatures
	auto features = std::make_shared<DenseFeatures<float64_t>>(matrix);

	// Construct kernel and compute kernel matrix
	auto kernel = std::make_shared<PeriodicKernel>(features, features, 1.0, 5.0);
	SGMatrix<float64_t> computed_kernel_matrix = kernel->get_kernel_matrix();

	// Define expected kernel matrix
	SGMatrix<float64_t> expected_kernel_matrix(3,3);
	expected_kernel_matrix(0,0) = 1.0;
	expected_kernel_matrix(1,0) = 0.14718930341788436;
	expected_kernel_matrix(2,0) = 0.724874286159619241;
	expected_kernel_matrix(0,1) = 0.14718930341788436;
	expected_kernel_matrix(1,1) = 1.0;
	expected_kernel_matrix(2,1) = 0.14718930341788436;
	expected_kernel_matrix(0,2) = 0.724874286159619241;
	expected_kernel_matrix(1,2) = 0.14718930341788436;
	expected_kernel_matrix(2,2) = 1.0;

	is_eqauls(expected_kernel_matrix, computed_kernel_matrix);

	// Clean up

}

TEST(PeriodicKernelTest,test_derivative_width)
{
	// Construct matrix with features
	SGMatrix<float64_t> matrix(2,3);
	for (int32_t i=0; i<6; i++) {
		matrix.matrix[i]=i;
	}

	// Load them into DenseFeatures
	auto features = std::make_shared<DenseFeatures<float64_t>>(matrix);

	// Construct kernel
	auto kernel = std::make_shared<PeriodicKernel>(features, features, 1.0, 5.0);

	// Compute derivative matrix
	Parameter *parameters = kernel->m_parameters;
	TParameter *width = parameters->get_parameter("length_scale");
	SGMatrix<float64_t> dMatrix = kernel->get_parameter_gradient(width);

	// Define expected derivative matrix
	SGMatrix<float64_t> expected_derivative_matrix(3,3);
	expected_derivative_matrix(0,0) = 0.0;
	expected_derivative_matrix(1,0) = 0.564039932473408001;
	expected_derivative_matrix(2,0) = 0.466466805840957677;
	expected_derivative_matrix(0,1) = 0.564039932473408001;
	expected_derivative_matrix(1,1) = 0.0;
	expected_derivative_matrix(2,1) = 0.564039932473408001;
	expected_derivative_matrix(0,2) = 0.466466805840957677;
	expected_derivative_matrix(1,2) = 0.564039932473408001;
	expected_derivative_matrix(2,2) = 0.0;

	is_eqauls(expected_derivative_matrix, dMatrix);

	// Clean up

}

TEST(PeriodicKernelTest,test_derivative_period)
{
	// Construct matrix with features
	SGMatrix<float64_t> matrix(2,3);
	for (int32_t i=0; i<6; i++) {
		matrix.matrix[i]=i;
	}

	// Load them into DenseFeatures
	auto features= std::make_shared<DenseFeatures<float64_t>>(matrix);

	// Construct kernel
	auto kernel = std::make_shared<PeriodicKernel>(features, features, 1.0, 5.0);

	// Compute derivative matrix
	Parameter *parameters = kernel->m_parameters;
	TParameter *period = parameters->get_parameter("period");
	SGMatrix<float64_t> dMatrix = kernel->get_parameter_gradient(period);

	// Define expected derivative matrix
	SGMatrix<float64_t> expected_derivative_matrix(3,3);
	expected_derivative_matrix(0,0) = 0.0;
	expected_derivative_matrix(1,0) = -0.0419672133442629894;
	expected_derivative_matrix(2,0) = 0.757301797430070978;
	expected_derivative_matrix(0,1) = -0.0419672133442629894;
	expected_derivative_matrix(1,1) = 0.0;
	expected_derivative_matrix(2,1) = -0.0419672133442629894;
	expected_derivative_matrix(0,2) = 0.757301797430070978;
	expected_derivative_matrix(1,2) = -0.0419672133442629894;
	expected_derivative_matrix(2,2) = 0.0;

	is_eqauls(expected_derivative_matrix, dMatrix);

	// Clean up

}
