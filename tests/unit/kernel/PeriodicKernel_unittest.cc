#include <shogun/kernel/PeriodicKernel.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/features/DenseFeatures.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(PeriodicKernelTest,test_kernel_matrix)
{
	// Construct matrix with features
	SGMatrix<float64_t> matrix(2,3);
	for (int32_t i=0; i<6; i++) {
		matrix.matrix[i]=i;
	}

	// Load them into DenseFeatures
	CDenseFeatures<float64_t>* features= new CDenseFeatures<float64_t>();
	features->set_feature_matrix(matrix);

	// Construct kernel and compute kernel matrix
	CPeriodicKernel* kernel = new CPeriodicKernel(features, features, 1.0, 5.0);
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

	EXPECT_EQ(true, computed_kernel_matrix.equals(expected_kernel_matrix));

	// Clean up
	SG_UNREF(kernel);
}

TEST(PeriodicKernelTest,test_derivative_width)
{
	// Construct matrix with features
	SGMatrix<float64_t> matrix(2,3);
	for (int32_t i=0; i<6; i++) {
		matrix.matrix[i]=i;
	}

	// Load them into DenseFeatures
	CDenseFeatures<float64_t>* features= new CDenseFeatures<float64_t>();
	features->set_feature_matrix(matrix);

	// Construct kernel
	CPeriodicKernel* kernel = new CPeriodicKernel(features, features, 1.0, 5.0);

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

	EXPECT_EQ(true, dMatrix.equals(expected_derivative_matrix));

	// Clean up
	SG_UNREF(kernel);
}

TEST(PeriodicKernelTest,test_derivative_period)
{
	// Construct matrix with features
	SGMatrix<float64_t> matrix(2,3);
	for (int32_t i=0; i<6; i++) {
		matrix.matrix[i]=i;
	}

	// Load them into DenseFeatures
	CDenseFeatures<float64_t>* features= new CDenseFeatures<float64_t>();
	features->set_feature_matrix(matrix);

	// Construct kernel
	CPeriodicKernel* kernel = new CPeriodicKernel(features, features, 1.0, 5.0);

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

	EXPECT_EQ(true, dMatrix.equals(expected_derivative_matrix));

	// Clean up
	SG_UNREF(kernel);
}
