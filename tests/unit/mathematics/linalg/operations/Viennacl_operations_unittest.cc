#include <shogun/lib/config.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <gtest/gtest.h>

#ifdef HAVE_VIENNACL
#include <shogun/mathematics/linalg/LinalgBackendViennaCL.h>

using namespace shogun;
using namespace linalg;

TEST(LinalgBackendViennaCL, SGVector_add)
{
	sg_linalg->set_gpu_backend(new LinalgBackendViennaCL());

	const float32_t alpha = 0.3;
	const float32_t beta = -1.5;

	SGVector<float32_t> A(9), A_gpu;
	SGVector<float32_t> B(9), B_gpu;

	for (index_t i = 0; i < 9; ++i)
	{
		A[i] = i;
		B[i] = 0.5*i;
	}

	A_gpu = to_gpu(A);
	B_gpu = to_gpu(B);

	auto result_gpu = add(A_gpu, B_gpu, alpha, beta);
	auto result = from_gpu(result_gpu);

	for (index_t i = 0; i < 9; ++i)
		EXPECT_NEAR(alpha*A[i]+beta*B[i], result[i], 1e-15);
}

TEST(LinalgBackendViennaCL, SGMatrix_add)
{
	sg_linalg->set_gpu_backend(new LinalgBackendViennaCL());

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
	A_gpu = to_gpu(A);
	B_gpu = to_gpu(B);

	auto result_gpu = add(A_gpu, B_gpu, alpha, beta);
	auto result = from_gpu(result_gpu);

	for (index_t i = 0; i < nrows*ncols; ++i)
		EXPECT_NEAR(alpha*A[i]+beta*B[i], result[i], 1e-15);
}

TEST(LinalgBackendViennaCL, SGVector_add_in_place)
{
	sg_linalg->set_gpu_backend(new LinalgBackendViennaCL());

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
	A_gpu = to_gpu(A);
	B_gpu = to_gpu(B);

	add(A_gpu, B_gpu, A_gpu, alpha, beta);
	A = from_gpu(A_gpu);

	for (index_t i = 0; i < 9; ++i)
		EXPECT_NEAR(alpha*C[i]+beta*B[i], A[i], 1e-15);
}

TEST(LinalgBackendViennaCL, SGMatrix_add_in_place)
{
	sg_linalg->set_gpu_backend(new LinalgBackendViennaCL());

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
	A_gpu = to_gpu(A);
	B_gpu = to_gpu(B);

	add(A_gpu, B_gpu, A_gpu, alpha, beta);
	A = from_gpu(A_gpu);

	for (index_t i = 0; i < nrows*ncols; ++i)
		EXPECT_NEAR(alpha*C[i]+beta*B[i], A[i], 1e-15);
}

TEST(LinalgBackendViennaCL, SGVector_dot)
{
	sg_linalg->set_gpu_backend(new LinalgBackendViennaCL());

	const index_t size = 3;
	SGVector<int32_t> a(size), b(size), a_gpu, b_gpu;
	a.range_fill(0);
	b.range_fill(0);

	a_gpu = to_gpu(a);
	b_gpu = to_gpu(b);

	auto result = dot(a_gpu, b_gpu);

	EXPECT_NEAR(result, 5, 1E-15);
}

TEST(LinalgBackendViennaCL, SGMatrix_elementwise_product)
{
	sg_linalg->set_gpu_backend(new LinalgBackendViennaCL());

	SGMatrix<float64_t> A(3,3);
	SGMatrix<float64_t> B(3,3);
	SGMatrix<float64_t> A_gpu, B_gpu;

	for (index_t i = 0; i < 9; ++i)
	{
		A[i] = i;
		B[i] = 0.5*i;
	}

	A_gpu = to_gpu(A);
	B_gpu = to_gpu(B);
	auto result_gpu = element_prod(A_gpu, B_gpu);
	auto result = from_gpu(result_gpu);

	for (index_t i = 0; i < 9; ++i)
		EXPECT_NEAR(A[i]*B[i], result[i], 1e-15);
}

TEST(LinalgBackendViennaCL, SGMatrix_elementwise_product_in_place)
{
	sg_linalg->set_gpu_backend(new LinalgBackendViennaCL());

	SGMatrix<float64_t> A(3,3);
	SGMatrix<float64_t> B(3,3);
	SGMatrix<float64_t> C(3,3);
	SGMatrix<float64_t> A_gpu, B_gpu;

	for (index_t i = 0; i < 9; ++i)
	{
		A[i] = i;
		B[i] = 0.5*i;
		C[i] = i;
	}

	A_gpu = to_gpu(A);
	B_gpu = to_gpu(B);
	element_prod(A_gpu, B_gpu, A_gpu);
	A = from_gpu(A_gpu);

	for (index_t i = 0; i < 9; ++i)
		EXPECT_NEAR(C[i]*B[i], A[i], 1e-15);
}

TEST(LinalgBackendViennaCL, SGMatrix_SGVector_matrix_prod)
{
	const index_t rows=4;
	const index_t cols=3;

	SGMatrix<float64_t> A(rows, cols), A_gpu;
	SGVector<float64_t> b(cols), b_gpu;

	for (index_t i = 0; i < cols; ++i)
	{
		for (index_t j = 0; j < rows; ++j)
			A(j, i) = i * rows + j;
		b[i] = 0.5 * i;
	}

	A_gpu = to_gpu(A);
	b_gpu = to_gpu(b);
	auto x_gpu = matrix_prod(A_gpu, b_gpu);
	auto x = from_gpu(x_gpu);

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

	for (index_t i = 0; i < cols; ++i)
	{
		for (index_t j = 0; j < rows; ++j)
			A(i, j) = i * cols + j;
		b[i] = 0.5 * i;
	}

	A_gpu = to_gpu(A);
	b_gpu = to_gpu(b);
	auto x_gpu = matrix_prod(A_gpu, b_gpu, true);
	auto x = from_gpu(x_gpu);

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
	x_gpu = to_gpu(x);

	A_gpu = to_gpu(A);
	b_gpu = to_gpu(b);
	matrix_prod(A_gpu, b_gpu, x_gpu);
	x = from_gpu(x_gpu);

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
	x_gpu = to_gpu(x);

	A_gpu = to_gpu(A);
	b_gpu = to_gpu(b);
	matrix_prod(A_gpu, b_gpu, x_gpu, true);
	x = from_gpu(x_gpu);

	float64_t ref[] = {7.5, 9, 10.5, 14.5};

	for (index_t i = 0; i < cols; ++i)
		EXPECT_NEAR(x[i], ref[i], 1e-15);
}

TEST(LinalgBackendViennaCL, SGMatrix_matrix_product)
{
	const index_t dim1 = 2, dim2 = 4, dim3 = 3;
	SGMatrix<float64_t> A(dim1, dim2);
	SGMatrix<float64_t> B(dim2, dim3);

	for (index_t i = 0; i < dim1*dim2; ++i)
		A[i] = i;
	for (index_t i = 0; i < dim2*dim3; ++i)
		B[i] = 2*i;

	SGMatrix<float64_t> A_gpu = to_gpu(A);
	SGMatrix<float64_t> B_gpu = to_gpu(B);
	auto cal_gpu = matrix_prod(A_gpu, B_gpu);
	auto cal = from_gpu(cal_gpu);

	int32_t ref[] = {56, 68, 152, 196, 248, 324};

	EXPECT_EQ(dim1, cal.num_rows);
	EXPECT_EQ(dim3, cal.num_cols);
	for (index_t i = 0; i < dim1*dim3; ++i)
		EXPECT_EQ(ref[i], cal[i]);
}

TEST(LinalgBackendViennaCL, SGMatrix_matrix_product_in_place)
{
	const index_t dim1 = 2, dim2 = 4, dim3 = 3;
	SGMatrix<float64_t> A(dim1, dim2);
	SGMatrix<float64_t> B(dim2, dim3);
	SGMatrix<float64_t> cal(dim1, dim3);

	for (index_t i = 0; i < dim1*dim2; ++i)
		A[i] = i;
	for (index_t i = 0; i < dim2*dim3; ++i)
		B[i] = 2*i;
	cal.zero();

	SGMatrix<float64_t> A_gpu = to_gpu(A);
	SGMatrix<float64_t> B_gpu = to_gpu(B);
	SGMatrix<float64_t>  cal_gpu = to_gpu(cal);
	linalg::matrix_prod(A_gpu, B_gpu, cal_gpu);
	cal = from_gpu(cal_gpu);

	int32_t ref[] = {14, 17, 38, 49, 62, 81};

	EXPECT_EQ(dim1, cal.num_rows);
	EXPECT_EQ(dim3, cal.num_cols);
	for (index_t i = 0; i < dim1*dim3; ++i)
		EXPECT_EQ(ref[i], cal[i]);
}

TEST(LinalgBackendViennaCL, SGVector_mean)
{
	sg_linalg->set_gpu_backend(new LinalgBackendViennaCL());

	const index_t size = 6;
	SGVector<int32_t> vec(size);
	SGVector<int32_t> vec_gpu;

	vec.range_fill(0);
	vec_gpu = to_gpu(vec);

	auto result = mean(vec_gpu);

	EXPECT_NEAR(result, 2.5, 1E-15);
}

TEST(LinalgBackendViennaCL, SGVector_max)
{
	sg_linalg->set_gpu_backend(new LinalgBackendViennaCL());

	SGVector<float32_t> A(9);
	SGVector<float32_t> A_gpu;

	float32_t a[] = {1, 2, 5, 8, 3, 1, 0, -1, 4};
	for (index_t i = 0; i < A.size(); ++i)
		A[i] = a[i];

	A_gpu = to_gpu(A);
	EXPECT_NEAR(8, max(A_gpu), 1e-15);
}

TEST(LinalgBackendViennaCL, SGMatrix_max)
{
	sg_linalg->set_gpu_backend(new LinalgBackendViennaCL());

	const index_t nrows = 2, ncols = 3;
	SGMatrix<float32_t> A(nrows, ncols);
	SGMatrix<float32_t> A_gpu;

	float32_t a[] = {1, 2, 5, 8, 3, 1, 0, -1, 12};
	for (index_t i = 0; i < nrows*ncols; ++i)
		A[i] = a[i];

	A_gpu = to_gpu(A);
	EXPECT_NEAR(8, max(A_gpu), 1e-15);
}

TEST(LinalgBackendViennaCL, SGMatrix_mean)
{
	sg_linalg->set_gpu_backend(new LinalgBackendViennaCL());

	const index_t nrows = 2;
	const index_t ncols = 3;
	SGMatrix<int32_t> mat(nrows, ncols);
	SGMatrix<int32_t> mat_gpu(nrows, ncols);
	for (index_t i = 0; i < nrows * ncols; ++i)
		mat[i] = i;

	mat_gpu = to_gpu(mat);

	auto result = mean(mat_gpu);
	EXPECT_NEAR(result, 2.5, 1E-15);
}

TEST(LinalgBackendViennaCL, SGVector_scale)
{
	sg_linalg->set_gpu_backend(new LinalgBackendViennaCL());

	const index_t size = 5;
	const float32_t alpha = 0.3;
	SGVector<float32_t> a(size), a_gpu;
	a.range_fill(0);

	a_gpu = to_gpu(a);

	SGVector<float32_t> result_gpu = scale(a_gpu, alpha);
	SGVector<float32_t> result = from_gpu(result_gpu);

	for (index_t i = 0; i < size; ++i)
		EXPECT_NEAR(alpha * a[i], result[i], 1e-15);
}

TEST(LinalgBackendViennaCL, SGMatrix_scale)
{
	sg_linalg->set_gpu_backend(new LinalgBackendViennaCL());

	const float32_t alpha = 0.3;
	const index_t nrows = 2, ncols = 3;
	SGMatrix<float32_t> A(nrows, ncols), A_gpu;
	for (index_t i = 0; i < nrows*ncols; ++i)
		A[i] = i;

	A_gpu = to_gpu(A);

	auto result_gpu = scale(A_gpu, alpha);
	auto result = from_gpu(result_gpu);

	for (index_t i = 0; i < nrows*ncols; ++i)
		EXPECT_NEAR(alpha*A[i], result[i], 1e-15);
}

TEST(LinalgBackendViennaCL, SGVector_scale_in_place)
{
	sg_linalg->set_gpu_backend(new LinalgBackendViennaCL());

	const index_t size = 5;
	const float32_t alpha = 0.3;
	SGVector<float32_t> a(size), a_gpu;
	a.range_fill(0);

	a_gpu = to_gpu(a);

	scale(a_gpu, a_gpu, alpha);
	a = from_gpu(a_gpu);

	for (index_t i = 0; i < size; ++i)
		EXPECT_NEAR(alpha * i, a[i], 1e-15);
}

TEST(LinalgBackendViennaCL, SGMatrix_scale_in_place)
{
	sg_linalg->set_gpu_backend(new LinalgBackendViennaCL());

	const float32_t alpha = 0.3;
	const index_t nrows = 2, ncols = 3;
	SGMatrix<float32_t> A(nrows, ncols);
	SGMatrix<float32_t> A_gpu;

	for (index_t i = 0; i < nrows*ncols; ++i)
		A[i] = i;

	A_gpu = to_gpu(A);

	scale(A_gpu, A_gpu, alpha);
	A = from_gpu(A_gpu);

	for (index_t i = 0; i < nrows*ncols; ++i)
		EXPECT_NEAR(alpha*i, A[i], 1e-15);
}

TEST(LinalgBackendViennaCL, SGVector_set_const)
{
	sg_linalg->set_gpu_backend(new LinalgBackendViennaCL());

	const index_t size = 5;
	const float32_t value = 2;
	SGVector<float32_t> a(size), a_gpu;

	a.range_fill(0);
	a_gpu = to_gpu(a);
	set_const(a_gpu, value);
	a = from_gpu(a_gpu);

	for (index_t i = 0; i < size; ++i)
		EXPECT_NEAR(a[i], value, 1E-15);
}

TEST(LinalgBackendViennaCL, SGMatrix_set_const)
{
	sg_linalg->set_gpu_backend(new LinalgBackendViennaCL());

	const index_t nrows = 2, ncols = 3;
	const float64_t value = 2;
	SGMatrix<float64_t> a(nrows, ncols), a_gpu;

	for (index_t i = 0; i < nrows*ncols; ++i)
		a[i] = i;

	a_gpu = to_gpu(a);
	set_const(a_gpu, value);
	a = from_gpu(a_gpu);

	for (index_t i = 0; i < nrows*ncols; ++i)
		EXPECT_NEAR(a[i], value, 1E-15);
}

TEST(LinalgBackendViennaCL, SGVector_sum)
{
	sg_linalg->set_gpu_backend(new LinalgBackendViennaCL());

	const index_t size = 10;
	SGVector<int32_t> vec(size), vec_gpu;
	vec.range_fill(0);
	vec_gpu = to_gpu(vec);

	auto result = sum(vec);
	EXPECT_NEAR(result, 45, 1E-15);
}

TEST(LinalgBackendViennaCL, SGMatrix_sum)
{
	sg_linalg->set_gpu_backend(new LinalgBackendViennaCL());

	const index_t nrows = 2, ncols = 3;
	SGMatrix<int32_t> a(nrows, ncols), a_gpu(nrows, ncols);
	for (index_t i = 0; i < nrows * ncols; ++i)
		a[i] = i;

	a_gpu = to_gpu(a);

	auto result = sum(a_gpu);
	EXPECT_NEAR(result, 15, 1E-15);
}

TEST(LinalgBackendViennaCL, SGMatrix_sum_no_diag)
{
	sg_linalg->set_gpu_backend(new LinalgBackendViennaCL());

	const index_t nrows = 2, ncols = 3;
	SGMatrix<int32_t> a(nrows, ncols), a_gpu(nrows, ncols);
	for (index_t i = 0; i < nrows * ncols; ++i)
		a[i] = i;

	a_gpu = to_gpu(a);

	auto result = sum(a_gpu, true);
	EXPECT_NEAR(result, 12, 1E-15);
}

TEST(LinalgBackendViennaCL, SGMatrix_colwise_sum)
{
	sg_linalg->set_gpu_backend(new LinalgBackendViennaCL());

	const index_t nrows = 2, ncols = 3;
	SGMatrix<int32_t> mat(nrows, ncols), mat_gpu;

	for (index_t i = 0; i < nrows * ncols; ++i)
		mat[i] = i;

	mat_gpu = to_gpu(mat);
	SGVector<int32_t> result_gpu = colwise_sum(mat_gpu);
	SGVector<int32_t> result = from_gpu(result_gpu);

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
	sg_linalg->set_gpu_backend(new LinalgBackendViennaCL());

	const index_t nrows = 2, ncols = 3;
	SGMatrix<int32_t> mat(nrows, ncols), mat_gpu;

	for (index_t i = 0; i < nrows * ncols; ++i)
		mat[i] = i;

	mat_gpu = to_gpu(mat);
	SGVector<int32_t> result_gpu = colwise_sum(mat_gpu, true);
	SGVector<int32_t> result = from_gpu(result_gpu);

	EXPECT_NEAR(result[0], 1, 1E-15);
	EXPECT_NEAR(result[1], 2, 1E-15);
	EXPECT_NEAR(result[2], 9, 1E-15);
}

TEST(LinalgBackendViennaCL, SGMatrix_rowwise_sum)
{
	sg_linalg->set_gpu_backend(new LinalgBackendViennaCL());

	const index_t nrows = 2, ncols = 3;
	SGMatrix<int32_t> mat(nrows, ncols), mat_gpu;

	for (index_t i = 0; i < nrows * ncols; ++i)
		mat[i] = i;

	mat_gpu = to_gpu(mat);
	SGVector<int32_t> result_gpu = rowwise_sum(mat_gpu);
	SGVector<int32_t> result = from_gpu(result_gpu);

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
	sg_linalg->set_gpu_backend(new LinalgBackendViennaCL());

	const index_t nrows = 2, ncols = 3;
	SGMatrix<int32_t> mat(nrows, ncols), mat_gpu;

	for (index_t i = 0; i < nrows * ncols; ++i)
		mat[i] = i;

	mat_gpu = to_gpu(mat);
	SGVector<int32_t> result_gpu = rowwise_sum(mat_gpu, true);
	SGVector<int32_t> result = from_gpu(result_gpu);

	EXPECT_NEAR(result[0], 6, 1E-15);
	EXPECT_NEAR(result[1], 6, 1E-15);
}

#endif // HAVE_VIENNACL
