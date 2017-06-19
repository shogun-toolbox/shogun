#include <shogun/lib/config.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/LinalgNamespaceMatrix.h>
#include <gtest/gtest.h>

using namespace shogun;
using namespace linalg;

template <typename T>
class LinalgBackendEigenMatrix : public ::testing::Test {};

typedef ::testing::Types<float32_t, float64_t, floatmax_t> PTypes;
TYPED_TEST_CASE(LinalgBackendEigenMatrix, PTypes);

TYPED_TEST(LinalgBackendEigenMatrix, add)
{
	const index_t nrows = 2, ncols = 3;

	TypeParam A_data[] = {0, 1, 2, 3, 4, 5};
	TypeParam B_data[] = {1, 2, 3, 4, 5, 6};
	TypeParam result_data[] = {0, 0, 0, 0, 0, 0};

	Matrix A(A_data, nrows, ncols);
	Matrix B(B_data, nrows, ncols);
	Matrix result(result_data, nrows, ncols);

	add(A, B, result);

	for (index_t i = 0; i < nrows*ncols; ++i)
		EXPECT_NEAR(A_data[i]+B_data[i], result.raw_data<TypeParam>()[i], 1e-15);
}