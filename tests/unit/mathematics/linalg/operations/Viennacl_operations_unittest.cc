#include <gtest/gtest.h>

#include <shogun/base/range.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/config.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/mathematics/linalg/LinalgSpecialPurposes.h>

#ifdef HAVE_VIENNACL
#include <shogun/mathematics/linalg/LinalgBackendViennaCL.h>

using namespace shogun;
using namespace linalg;

TEST(LinalgBackendViennaCL, SGVector_add)
{
	env()->linalg()->set_gpu_backend(new LinalgBackendViennaCL());

	const float32_t alpha = 0.3;
	const float32_t beta = -1.5;

	SGVector<float32_t> A(9), A_gpu;
	SGVector<float32_t> B(9), B_gpu;
	SGVector<float32_t> result;

	for (index_t i = 0; i < 9; ++i)
	{
		A[i] = i;
		B[i] = 0.5*i;
	}

	to_gpu(A, A_gpu);
	to_gpu(B, B_gpu);

	auto result_gpu = add(A_gpu, B_gpu, alpha, beta);
	from_gpu(result_gpu, result);

	for (index_t i = 0; i < 9; ++i)
		EXPECT_NEAR(alpha*A[i]+beta*B[i], result[i], 1e-15);
}

TEST(LinalgBackendViennaCL, SGMatrix_add)
{
	env()->linalg()->set_gpu_backend(new LinalgBackendViennaCL());

	const float32_t alpha = 0.3;
	const float32_t beta = -1.5;
	const index_t nrows = 2, ncols = 3;

	SGMatrix<float32_t> A(nrows, ncols), A_gpu;
	SGMatrix<float32_t> B(nrows, ncols), B_gpu;

	for (index_t i = 0; i < nrows*ncols; ++i)
	{
		A[i] = i;
		B[i] = 0.5*i;
	}
	to_gpu(A, A_gpu);
	to_gpu(B, B_gpu);

	auto result_gpu = add(A_gpu, B_gpu, alpha, beta);
	SGMatrix<float32_t> result;
	from_gpu(result_gpu, result);

	for (index_t i = 0; i < nrows*ncols; ++i)
		EXPECT_NEAR(alpha*A[i]+beta*B[i], result[i], 1e-15);
}

TEST(LinalgBackendViennaCL, SGVector_add_in_place)
{
	env()->linalg()->set_gpu_backend(new LinalgBackendViennaCL());

	const float32_t alpha = 0.3;
	const float32_t beta = -1.5;

	SGVector<float32_t> A(9), B(9), C(9);
	SGVector<float32_t> A_gpu, B_gpu;

	for (index_t i = 0; i < 9; ++i)
	{
		A[i] = i;
		B[i] = 0.5*i;
		C[i] = i;
	}
	to_gpu(A, A_gpu);
	to_gpu(B, B_gpu);

	add(A_gpu, B_gpu, A_gpu, alpha, beta);
	from_gpu(A_gpu, A);

	for (index_t i = 0; i < 9; ++i)
		EXPECT_NEAR(alpha*C[i]+beta*B[i], A[i], 1e-15);
}

TEST(LinalgBackendViennaCL, SGMatrix_add_in_place)
{
	env()->linalg()->set_gpu_backend(new LinalgBackendViennaCL());

	const float32_t alpha = 0.3;
	const float32_t beta = -1.5;
	const index_t nrows = 2, ncols = 3;

	SGMatrix<float32_t> A(nrows, ncols), A_gpu;
	SGMatrix<float32_t> B(nrows, ncols), B_gpu;
	SGMatrix<float32_t> C(nrows, ncols);

	for (index_t i = 0; i < nrows*ncols; ++i)
	{
		A[i] = i;
		B[i] = 0.5*i;
		C[i] = i;
	}
	to_gpu(A, A_gpu);
	to_gpu(B, B_gpu);

	add(A_gpu, B_gpu, A_gpu, alpha, beta);
	from_gpu(A_gpu, A);

	for (index_t i = 0; i < nrows*ncols; ++i)
		EXPECT_NEAR(alpha*C[i]+beta*B[i], A[i], 1e-15);
}

TEST(LinalgBackendViennaCL, SGMatrix_cross_entropy)
{
	SGMatrix<float64_t> A(4, 3), A_gpu;
	SGMatrix<float64_t> B(4, 3), B_gpu;

	int32_t size = A.num_rows * A.num_cols;
	for (float64_t i = 0; i < size; ++i)
	{
		A[i] = i / size;
		B[i] = (i / size) * 0.5;
	}

	float64_t ref = 0;
	for (int32_t i = 0; i < size; i++)
		ref += A[i] * std::log(B[i] + 1e-30);
	ref *= -1;

	to_gpu(A, A_gpu);
	to_gpu(B, B_gpu);
	auto result = linalg::cross_entropy(A_gpu, B_gpu);

	EXPECT_NEAR(ref, result, 1e-15);
}

TEST(LinalgBackendViennaCL, SGVector_dot)
{
	env()->linalg()->set_gpu_backend(new LinalgBackendViennaCL());

	const index_t size = 3;
	SGVector<int32_t> a(size), b(size), a_gpu, b_gpu;
	a.range_fill(0);
	b.range_fill(0);

	to_gpu(a, a_gpu);
	to_gpu(b, b_gpu);

	auto result = dot(a_gpu, b_gpu);

	EXPECT_NEAR(result, 5, 1E-15);
}

TEST(LinalgBackendViennaCL, SGMatrix_elementwise_product)
{
	env()->linalg()->set_gpu_backend(new LinalgBackendViennaCL());

	const auto m = 3;
	SGMatrix<float64_t> A(m, m);
	SGMatrix<float64_t> B(m, m);
	SGMatrix<float64_t> A_gpu, B_gpu;
	SGMatrix<float64_t> result;

	for (auto i : range(m * m))
	{
		A[i] = i;
		B[i] = 0.5*i;
	}

	to_gpu(A, A_gpu);
	to_gpu(B, B_gpu);

	auto result_gpu = element_prod(A_gpu, B_gpu);
	from_gpu(result_gpu, result);
	for (auto i : range(m))
		for (auto j : range(m))
			EXPECT_NEAR(A(i, j) * B(i, j), result(i, j), 1e-15);

	result_gpu = element_prod(A_gpu, B_gpu, true, false);
	from_gpu(result_gpu, result);
	for (auto i : range(m))
		for (auto j : range(m))
			EXPECT_NEAR(A(j, i) * B(i, j), result(i, j), 1e-15);

	result_gpu = element_prod(A_gpu, B_gpu, true, false);
	from_gpu(result_gpu, result);
	for (auto i : range(m))
		for (auto j : range(m))
			EXPECT_NEAR(A(i, j) * B(j, i), result(i, j), 1e-15);

	result_gpu = element_prod(A_gpu, B_gpu, true, true);
	from_gpu(result_gpu, result);
	for (auto i : range(m))
		for (auto j : range(m))
			EXPECT_NEAR(A(j, i) * B(j, i), result(i, j), 1e-15);
}

TEST(LinalgBackendViennaCL, SGMatrix_elementwise_product_in_place)
{
	env()->linalg()->set_gpu_backend(new LinalgBackendViennaCL());

	const auto m = 3;
	SGMatrix<float64_t> A(m, m);
	SGMatrix<float64_t> B(m, m);
	SGMatrix<float64_t> result(m, m);
	SGMatrix<float64_t> A_gpu, B_gpu, result_gpu;

	for (auto i : range(m))
	{
		A[i] = i;
		B[i] = 0.5*i;
	}

	to_gpu(A, A_gpu);
	to_gpu(B, B_gpu);
	to_gpu(result, result_gpu);

	element_prod(A_gpu, B_gpu, result_gpu);
	from_gpu(result_gpu, result);
	for (auto i : range(m))
		for (auto j : range(m))
			EXPECT_NEAR(A(i, j) * B(i, j), result(i, j), 1e-15);

	element_prod(A_gpu, B_gpu, result_gpu, true, false);
	from_gpu(result_gpu, result);
	for (auto i : range(m))
		for (auto j : range(m))
			EXPECT_NEAR(A(j, i) * B(i, j), result(i, j), 1e-15);

	element_prod(A_gpu, B_gpu, result_gpu, false, true);
	from_gpu(result_gpu, result);
	for (auto i : range(m))
		for (auto j : range(m))
			EXPECT_NEAR(A(i, j) * B(j, i), result(i, j), 1e-15);

	element_prod(A_gpu, B_gpu, result_gpu, true, true);
	from_gpu(result_gpu, result);
	for (auto i : range(m))
		for (auto j : range(m))
			EXPECT_NEAR(A(j, i) * B(j, i), result(i, j), 1e-15);
}

TEST(LinalgBackendViennaCL, SGMatrix_logistic)
{
	SGMatrix<float64_t> A(3,3), A_gpu;
	SGMatrix<float64_t> B(3,3), B_gpu;

	range_fill(A, 0.0);
	B.zero();

	to_gpu(A, A_gpu);
	to_gpu(B, B_gpu);

	linalg::logistic(A_gpu, B_gpu);

	from_gpu(A_gpu, A);
	from_gpu(B_gpu, B);

	for (index_t i = 0; i < 9; ++i)
		EXPECT_NEAR(1.0 / (1 + std::exp(-1 * A[i])), B[i], 1e-15);
}

TEST(LinalgBackendViennaCL, SGMatrix_SGVector_matrix_prod)
{
	const index_t rows=4;
	const index_t cols=3;

	SGMatrix<float64_t> A(rows, cols), A_gpu;
	SGVector<float64_t> b(cols), b_gpu;
	SGVector<float64_t> x;

	for (index_t i = 0; i < cols; ++i)
	{
		for (index_t j = 0; j < rows; ++j)
			A(j, i) = i * rows + j;
		b[i] = 0.5 * i;
	}

	to_gpu(A, A_gpu);
	to_gpu(b, b_gpu);
	auto x_gpu = matrix_prod(A_gpu, b_gpu);
	from_gpu(x_gpu, x);

	float64_t ref[] = {10, 11.5, 13, 14.5};

	EXPECT_EQ(x.vlen, A.num_rows);
	for (index_t i = 0; i < cols; ++i)
		EXPECT_NEAR(x[i], ref[i], 1e-15);
}

TEST(LinalgBackendViennaCL, SGMatrix_SGVector_matrix_prod_transpose)
{
	const index_t rows=4;
	const index_t cols=3;

	SGMatrix<float64_t> A(cols, rows), A_gpu;
	SGVector<float64_t> b(cols), b_gpu;
	SGVector<float64_t> x;

	for (index_t i = 0; i < cols; ++i)
	{
		for (index_t j = 0; j < rows; ++j)
			A(i, j) = i * cols + j;
		b[i] = 0.5 * i;
	}

	to_gpu(A, A_gpu);
	to_gpu(b, b_gpu);
	auto x_gpu = matrix_prod(A_gpu, b_gpu, true);
	from_gpu(x_gpu, x);

	float64_t ref[] = {7.5, 9, 10.5, 14.5};

	EXPECT_EQ(x.vlen, A.num_cols);
	for (index_t i = 0; i < cols; ++i)
		EXPECT_NEAR(x[i], ref[i], 1e-15);
}

TEST(LinalgBackendViennaCL, SGMatrix_SGVector_matrix_prod_in_place)
{
	const index_t rows=4;
	const index_t cols=3;

	SGMatrix<float64_t> A(rows, cols), A_gpu;
	SGVector<float64_t> b(cols), b_gpu;
	SGVector<float64_t> x(rows), x_gpu;

	for (index_t i = 0; i < cols; ++i)
	{
		for (index_t j = 0; j < rows; ++j)
			A(j, i) = i * rows + j;
		b[i] = 0.5 * i;
	}
	x.zero();
	to_gpu(x, x_gpu);

	to_gpu(A, A_gpu);
	to_gpu(b, b_gpu);
	matrix_prod(A_gpu, b_gpu, x_gpu);
	from_gpu(x_gpu, x);

	float64_t ref[] = {10, 11.5, 13, 14.5};

	EXPECT_EQ(x.vlen, A.num_rows);
	for (index_t i=0; i<cols; ++i)
		EXPECT_NEAR(x[i], ref[i], 1e-15);
}

TEST(LinalgBackendViennaCL, SGMatrix_SGVector_matrix_prod_in_place_transpose)
{
	const index_t rows=4;
	const index_t cols=3;

	SGMatrix<float64_t> A(cols, rows), A_gpu;
	SGVector<float64_t> b(cols), b_gpu;
	SGVector<float64_t> x(rows), x_gpu;

	for (index_t i = 0; i < cols; ++i)
	{
		for (index_t j = 0; j < rows; ++j)
			A(i, j) = i * cols + j;
		b[i] = 0.5 * i;
	}

	x.zero();
	to_gpu(x, x_gpu);

	to_gpu(A, A_gpu);
	to_gpu(b, b_gpu);
	matrix_prod(A_gpu, b_gpu, x_gpu, true);
	from_gpu(x_gpu, x);

	float64_t ref[] = {7.5, 9, 10.5, 14.5};

	for (index_t i = 0; i < cols; ++i)
		EXPECT_NEAR(x[i], ref[i], 1e-15);
}

TEST(LinalgBackendViennaCL, SGMatrix_matrix_product)
{
	const index_t dim1 = 2, dim2 = 4, dim3 = 3;
	SGMatrix<float64_t> A(dim1, dim2), A_gpu;
	SGMatrix<float64_t> B(dim2, dim3), B_gpu;

	for (index_t i = 0; i < dim1*dim2; ++i)
		A[i] = i;
	for (index_t i = 0; i < dim2*dim3; ++i)
		B[i] = 2*i;

	to_gpu(A, A_gpu);
	to_gpu(B, B_gpu);

	auto cal_gpu = matrix_prod(A_gpu, B_gpu);
	SGMatrix<float64_t> cal;
	from_gpu(cal_gpu, cal);

	int32_t ref[] = {56, 68, 152, 196, 248, 324};

	EXPECT_EQ(dim1, cal.num_rows);
	EXPECT_EQ(dim3, cal.num_cols);
	for (index_t i = 0; i < dim1*dim3; ++i)
		EXPECT_EQ(ref[i], cal[i]);
}

TEST(LinalgBackendViennaCL, SGMatrix_matrix_product_in_place)
{
	const index_t dim1 = 2, dim2 = 4, dim3 = 3;
	SGMatrix<float64_t> A(dim1, dim2), A_gpu;
	SGMatrix<float64_t> B(dim2, dim3), B_gpu;
	SGMatrix<float64_t> cal(dim1, dim3), cal_gpu;

	for (index_t i = 0; i < dim1*dim2; ++i)
		A[i] = i;
	for (index_t i = 0; i < dim2*dim3; ++i)
		B[i] = 0.5 * i;
	cal.zero();

	to_gpu(A, A_gpu);
	to_gpu(B, B_gpu);
	to_gpu(cal, cal_gpu);

	linalg::matrix_prod(A_gpu, B_gpu, cal_gpu);
	from_gpu(cal_gpu, cal);

	int32_t ref[] = {14, 17, 38, 49, 62, 81};

	EXPECT_EQ(dim1, cal.num_rows);
	EXPECT_EQ(dim3, cal.num_cols);
	for (index_t i = 0; i < dim1*dim3; ++i)
		EXPECT_EQ(ref[i], cal[i]);
}

TEST(LinalgBackendViennaCL, SGVector_mean)
{
	env()->linalg()->set_gpu_backend(new LinalgBackendViennaCL());

	const index_t size = 6;
	SGVector<int32_t> vec(size);
	SGVector<int32_t> vec_gpu;

	vec.range_fill(0);
	to_gpu(vec, vec_gpu);

	auto result = mean(vec_gpu);

	EXPECT_NEAR(result, 2.5, 1E-15);
}

TEST(LinalgBackendViennaCL, SGVector_max)
{
	env()->linalg()->set_gpu_backend(new LinalgBackendViennaCL());

	SGVector<float32_t> A(9);
	SGVector<float32_t> A_gpu;

	float32_t a[] = {1, 2, 5, 8, 3, 1, 0, -1, 4};
	for (index_t i = 0; i < A.size(); ++i)
		A[i] = a[i];

	to_gpu(A, A_gpu);
	EXPECT_NEAR(8, max(A_gpu), 1e-15);
}

TEST(LinalgBackendViennaCL, SGMatrix_max)
{
	env()->linalg()->set_gpu_backend(new LinalgBackendViennaCL());

	const index_t nrows = 2, ncols = 3;
	SGMatrix<float32_t> A(nrows, ncols);
	SGMatrix<float32_t> A_gpu;

	float32_t a[] = {1, 2, 5, 8, 3, 1, 0, -1, 12};
	for (index_t i = 0; i < nrows*ncols; ++i)
		A[i] = a[i];

	to_gpu(A, A_gpu);
	EXPECT_NEAR(8, max(A_gpu), 1e-15);
}

TEST(LinalgBackendViennaCL, SGMatrix_mean)
{
	env()->linalg()->set_gpu_backend(new LinalgBackendViennaCL());

	const index_t nrows = 2;
	const index_t ncols = 3;
	SGMatrix<int32_t> mat(nrows, ncols);
	SGMatrix<int32_t> mat_gpu(nrows, ncols);
	for (index_t i = 0; i < nrows * ncols; ++i)
		mat[i] = i;

	to_gpu(mat, mat_gpu);

	auto result = mean(mat_gpu);
	EXPECT_NEAR(result, 2.5, 1E-15);
}

TEST(LinalgBackendViennaCL, SGMatrix_multiply_by_logistic_derivative)
{
	SGMatrix<float64_t> A(3, 3), A_gpu;
	SGMatrix<float64_t> B(3, 3), B_gpu;

	for (float64_t i = 0; i < 9; ++i)
	{
		A[i] = i / 9;
		B[i] = i;
	}

	to_gpu(A, A_gpu);
	to_gpu(B, B_gpu);

	linalg::multiply_by_logistic_derivative(A_gpu, B_gpu);

	from_gpu(A_gpu, A);
	from_gpu(B_gpu, B);

	for (index_t i = 0; i < 9; ++i)
		EXPECT_NEAR(i * A[i] * (1.0 - A[i]), B[i], 1e-15);
}

TEST(LinalgBackendViennaCL, SGMatrix_multiply_by_rectified_linear_derivative)
{
	env()->linalg()->set_gpu_backend(new LinalgBackendViennaCL());

	SGMatrix<float64_t> A(3, 3), A_gpu;
	SGMatrix<float64_t> B(3, 3), B_gpu;

	for (float64_t i = 0; i < 9; ++i)
	{
		A[i] = i * 0.5 - 0.5;
		B[i] = i;
	}

	to_gpu(A, A_gpu);
	to_gpu(B, B_gpu);

	linalg::multiply_by_rectified_linear_derivative(A_gpu, B_gpu);

	from_gpu(A_gpu, A);
	from_gpu(B_gpu, B);

	for (index_t i = 0; i < 9; ++i)
		EXPECT_NEAR(i * (A[i] != 0), B[i], 1e-15);
}

TEST(LinalgBackendViennaCL, SGMatrix_rectified_linear)
{
	SGMatrix<float64_t> A(3, 3), A_gpu;
	SGMatrix<float64_t> B(3, 3), B_gpu;

	range_fill(A, -5.0);
	to_gpu(A, A_gpu);
	to_gpu(B, B_gpu);

	linalg::rectified_linear(A_gpu, B_gpu);

	from_gpu(A_gpu, A);
	from_gpu(B_gpu, B);
	for (index_t i = 0; i < 9; ++i)
		EXPECT_NEAR(Math::max(0.0, A[i]), B[i], 1e-15);
}

TEST(LinalgBackendViennaCL, SGVector_scale)
{
	env()->linalg()->set_gpu_backend(new LinalgBackendViennaCL());

	const index_t size = 5;
	const float32_t alpha = 0.3;
	SGVector<float32_t> a(size), a_gpu;
	a.range_fill(0);

	to_gpu(a, a_gpu);

	SGVector<float32_t> result_gpu = scale(a_gpu, alpha);
	SGVector<float32_t> result;
	from_gpu(result_gpu, result);

	for (index_t i = 0; i < size; ++i)
		EXPECT_NEAR(alpha * a[i], result[i], 1e-15);
}

TEST(LinalgBackendViennaCL, SGMatrix_scale)
{
	env()->linalg()->set_gpu_backend(new LinalgBackendViennaCL());

	const float32_t alpha = 0.3;
	const index_t nrows = 2, ncols = 3;
	SGMatrix<float32_t> A(nrows, ncols), A_gpu;
	for (index_t i = 0; i < nrows*ncols; ++i)
		A[i] = i;

	to_gpu(A, A_gpu);

	auto result_gpu = scale(A_gpu, alpha);
	SGMatrix<float32_t> result;
	from_gpu(result_gpu, result);

	for (index_t i = 0; i < nrows*ncols; ++i)
		EXPECT_NEAR(alpha*A[i], result[i], 1e-15);
}

TEST(LinalgBackendViennaCL, SGVector_scale_in_place)
{
	env()->linalg()->set_gpu_backend(new LinalgBackendViennaCL());

	const index_t size = 5;
	const float32_t alpha = 0.3;
	SGVector<float32_t> a(size), a_gpu;
	a.range_fill(0);

	to_gpu(a, a_gpu);

	scale(a_gpu, a_gpu, alpha);
	from_gpu(a_gpu, a);

	for (index_t i = 0; i < size; ++i)
		EXPECT_NEAR(alpha * i, a[i], 1e-15);
}

TEST(LinalgBackendViennaCL, SGMatrix_scale_in_place)
{
	env()->linalg()->set_gpu_backend(new LinalgBackendViennaCL());

	const float32_t alpha = 0.3;
	const index_t nrows = 2, ncols = 3;
	SGMatrix<float32_t> A(nrows, ncols);
	SGMatrix<float32_t> A_gpu;

	for (index_t i = 0; i < nrows*ncols; ++i)
		A[i] = i;

	to_gpu(A, A_gpu);

	scale(A_gpu, A_gpu, alpha);
	from_gpu(A_gpu, A);

	for (index_t i = 0; i < nrows*ncols; ++i)
		EXPECT_NEAR(alpha*i, A[i], 1e-15);
}

TEST(LinalgBackendViennaCL, SGVector_set_const)
{
	env()->linalg()->set_gpu_backend(new LinalgBackendViennaCL());

	const index_t size = 5;
	const float32_t value = 2;
	SGVector<float32_t> a(size), a_gpu;

	a.range_fill(0);
	to_gpu(a, a_gpu);
	set_const(a_gpu, value);
	from_gpu(a_gpu, a);

	for (index_t i = 0; i < size; ++i)
		EXPECT_NEAR(a[i], value, 1E-15);
}

TEST(LinalgBackendViennaCL, SGMatrix_set_const)
{
	env()->linalg()->set_gpu_backend(new LinalgBackendViennaCL());

	const index_t nrows = 2, ncols = 3;
	const float64_t value = 2;
	SGMatrix<float64_t> a(nrows, ncols), a_gpu;

	for (index_t i = 0; i < nrows*ncols; ++i)
		a[i] = i;

	to_gpu(a, a_gpu);
	set_const(a_gpu, value);
	from_gpu(a_gpu, a);

	for (index_t i = 0; i < nrows*ncols; ++i)
		EXPECT_NEAR(a[i], value, 1E-15);
}

TEST(LinalgBackendViennaCL, SGMatrix_softmax)
{
	SGMatrix<float64_t> A(4, 3), A_gpu;
	SGMatrix<float64_t> ref(4, 3);

	for (float64_t i = 0; i < 12; ++i)
		A[i] = i / 12;

	for (index_t i = 0; i < 12; ++i)
		ref[i] = std::exp(A[i]);

	for (index_t j = 0; j < ref.num_cols; ++j)
	{
		float64_t sum = 0;
		for (index_t i = 0; i < ref.num_rows; ++i)
			sum += ref(i, j);

		for (index_t i = 0; i < ref.num_rows; ++i)
			ref(i, j) /= sum;
	}

	to_gpu(A, A_gpu);
	linalg::softmax(A_gpu);
	from_gpu(A_gpu, A);

	for (index_t i = 0; i < 12; ++i)
		EXPECT_NEAR(ref[i], A[i], 1e-15);
}

TEST(LinalgBackendViennaCL, SGMatrix_squared_error)
{
	SGMatrix<float64_t> A(4, 3), A_gpu;
	SGMatrix<float64_t> B(4, 3), B_gpu;

	int32_t size = A.num_rows * A.num_cols;
	for (float64_t i = 0; i < size; ++i)
	{
		A[i] = i / size;
		B[i] = (i / size) * 0.5;
	}

	float64_t ref = 0;
	for (index_t i = 0; i < size; i++)
		ref += Math::pow(A[i] - B[i], 2);
	ref *= 0.5;

	to_gpu(A, A_gpu);
	to_gpu(B, B_gpu);
	auto result = linalg::squared_error(A_gpu, B_gpu);
	EXPECT_NEAR(ref, result, 1e-15);
}

TEST(LinalgBackendViennaCL, SGVector_sum)
{
	env()->linalg()->set_gpu_backend(new LinalgBackendViennaCL());

	const index_t size = 10;
	SGVector<int32_t> vec(size), vec_gpu;
	vec.range_fill(0);
	to_gpu(vec, vec_gpu);

	auto result = sum(vec);
	EXPECT_NEAR(result, 45, 1E-15);
}

TEST(LinalgBackendViennaCL, SGMatrix_sum)
{
	env()->linalg()->set_gpu_backend(new LinalgBackendViennaCL());

	const index_t nrows = 2, ncols = 3;
	SGMatrix<int32_t> a(nrows, ncols), a_gpu(nrows, ncols);
	for (index_t i = 0; i < nrows * ncols; ++i)
		a[i] = i;

	to_gpu(a, a_gpu);

	auto result = sum(a_gpu);
	EXPECT_NEAR(result, 15, 1E-15);
}

TEST(LinalgBackendViennaCL, SGMatrix_sum_no_diag)
{
	env()->linalg()->set_gpu_backend(new LinalgBackendViennaCL());

	const index_t nrows = 2, ncols = 3;
	SGMatrix<int32_t> a(nrows, ncols), a_gpu(nrows, ncols);
	for (index_t i = 0; i < nrows * ncols; ++i)
		a[i] = i;

	to_gpu(a, a_gpu);

	auto result = sum(a_gpu, true);
	EXPECT_NEAR(result, 12, 1E-15);
}

TEST(LinalgBackendViennaCL, SGMatrix_colwise_sum)
{
	env()->linalg()->set_gpu_backend(new LinalgBackendViennaCL());

	const index_t nrows = 2, ncols = 3;
	SGMatrix<int32_t> mat(nrows, ncols), mat_gpu;

	for (index_t i = 0; i < nrows * ncols; ++i)
		mat[i] = i;

	to_gpu(mat, mat_gpu);
	SGVector<int32_t> result_gpu = colwise_sum(mat_gpu);
	SGVector<int32_t> result;
	from_gpu(result_gpu, result);

	for (index_t j = 0; j < ncols; ++j)
	{
		int32_t sum = 0;
		for (index_t i = 0; i < nrows; ++i)
			sum += mat(i, j);
		EXPECT_NEAR(sum, result[j], 1E-15);
	}
}

TEST(LinalgBackendViennaCL, SGMatrix_colwise_sum_no_diag)
{
	env()->linalg()->set_gpu_backend(new LinalgBackendViennaCL());

	const index_t nrows = 2, ncols = 3;
	SGMatrix<int32_t> mat(nrows, ncols), mat_gpu;

	for (index_t i = 0; i < nrows * ncols; ++i)
		mat[i] = i;

	to_gpu(mat, mat_gpu);
	SGVector<int32_t> result_gpu = colwise_sum(mat_gpu, true);
	SGVector<int32_t> result;
	from_gpu(result_gpu, result);

	EXPECT_NEAR(result[0], 1, 1E-15);
	EXPECT_NEAR(result[1], 2, 1E-15);
	EXPECT_NEAR(result[2], 9, 1E-15);
}

TEST(LinalgBackendViennaCL, SGMatrix_rowwise_sum)
{
	env()->linalg()->set_gpu_backend(new LinalgBackendViennaCL());

	const index_t nrows = 2, ncols = 3;
	SGMatrix<int32_t> mat(nrows, ncols), mat_gpu;

	for (index_t i = 0; i < nrows * ncols; ++i)
		mat[i] = i;

	to_gpu(mat, mat_gpu);
	SGVector<int32_t> result_gpu = rowwise_sum(mat_gpu);
	SGVector<int32_t> result;
	from_gpu(result_gpu, result);

	for (index_t i = 0; i < nrows; ++i)
	{
		int32_t sum = 0;
		for (index_t j = 0; j < ncols; ++j)
			sum += mat(i, j);
		EXPECT_NEAR(sum, result[i], 1E-15);
	}
}

TEST(LinalgBackendViennaCL, SGMatrix_rowwise_sum_no_diag)
{
	env()->linalg()->set_gpu_backend(new LinalgBackendViennaCL());

	const index_t nrows = 2, ncols = 3;
	SGMatrix<int32_t> mat(nrows, ncols), mat_gpu;

	for (index_t i = 0; i < nrows * ncols; ++i)
		mat[i] = i;

	to_gpu(mat, mat_gpu);
	SGVector<int32_t> result_gpu = rowwise_sum(mat_gpu, true);
	SGVector<int32_t> result;
	from_gpu(result_gpu, result);

	EXPECT_NEAR(result[0], 6, 1E-15);
	EXPECT_NEAR(result[1], 6, 1E-15);
}

#endif // HAVE_VIENNACL
