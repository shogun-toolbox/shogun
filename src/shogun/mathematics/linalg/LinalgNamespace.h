/*
 * Copyright (c) 2016, Shogun-Toolbox e.V. <shogun-team@shogun-toolbox.org>
 * All rights reserved.
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Authors: 2016 Pan Deng, Soumyajit De, Heiko Strathmann, Viktor Gal
 */

#include <shogun/mathematics/linalg/LinalgBackendBase.h>
#include <shogun/mathematics/linalg/SGLinalg.h>

namespace shogun
{

namespace linalg
{

/** Infer the appropriate backend for linalg operations
 * from the input SGVector or SGMatrix(Container).
 *
 * @param a SGVector or SGMatrix
 * @return LinalgBackendBase pointer
 */
template <typename T, template <typename> class Container>
LinalgBackendBase* infer_backend(const Container<T>& a)
{
	if (a.on_gpu())
	{
		if (sg_linalg->get_gpu_backend())
			return sg_linalg->get_gpu_backend();
		else
		{
			SG_SERROR("Vector memory on GPU but no GPU backend registered. \
						This can happen if the GPU backend was de-activated \
						after memory has been transferred to GPU.\n");
			return NULL;
		}
	}
	else
		return sg_linalg->get_cpu_backend();
}

/** Infer the appropriate backend for linalg operations
 * from the input SGVector or SGMatrix(Container).
 * Raise error if the backends from the two Containers don't match.
 *
 * @param a the first SGVector/SGMatrix
 * @param b the second SGVector/SGMatrix
 * @return LinalgBackendBase pointer
 */
template <typename T, template <typename> class Container>
LinalgBackendBase* infer_backend(const Container<T>& a, const Container<T>& b)
{
	if (a.on_gpu() && b.on_gpu())
	{
		if (sg_linalg->get_gpu_backend())
			return sg_linalg->get_gpu_backend();
		else
		{
			SG_SERROR("Vector memory on GPU but no GPU backend registered. \
					  This can happen if the GPU backend was de-activated \
					  after memory has been transferred to GPU.\n");
			return NULL;
		}
	}
	else if (a.on_gpu() || b.on_gpu())
	{
		SG_SERROR("Cannot operate with first vector gpu (%d) and second vector gpu (%d).\n",
					a.on_gpu(), b.on_gpu());
		return NULL;
	}
	else
		return sg_linalg->get_cpu_backend();
}


/**
 * Transfers data to GPU memory. Does nothing if no GPU backend registered.
 *
 * @param vector SGVector to be transferred
 * @return SGVector with vector on GPU if GPU backend is available
 * and a shallow-copy of SGVector with vector on CPU if GPU backend not available
 */
template <typename T>
SGVector<T> to_gpu(const SGVector<T>& vector)
{
	REQUIRE(!vector.on_gpu(), "The vector is already on GPU.\n");
	LinalgBackendBase* gpu_backend = sg_linalg->get_gpu_backend();
	if (gpu_backend)
		return SGVector<T>(gpu_backend->to_gpu(vector), vector.vlen);
	else
	{
		SG_SWARNING("Trying to access GPU memory without GPU backend registered.\n");
		return vector;
	}
}

/**
 * Transfers data to GPU memory. Does nothing if no GPU backend registered.
 *
 * @param vector SGMatrix to be transferred
 * @return SGMatrix with matrix on GPU if GPU backend is available
 * and a shallow-copy of SGMatrix with matrix on CPU if GPU backend not available
 */
template <typename T>
SGMatrix<T> to_gpu(const SGMatrix<T>& mat)
{
	REQUIRE(!mat.on_gpu(), "The matrix is already on GPU.\n");
	LinalgBackendBase* gpu_backend = sg_linalg->get_gpu_backend();
	if (gpu_backend)
		return SGMatrix<T>(gpu_backend->to_gpu(mat), mat.num_rows, mat.num_cols);
	else
	{
		SG_SWARNING("Trying to access GPU memory without GPU backend registered.\n");
		return mat;
	}
}

/**
 * Fetches data from GPU memory.
 *
 * @param vector SGVector to be transferred
 * @return SGVector with vector on CPU if GPU backend is still available
 * and a shallow-copy of SGVector with vector on GPU if GPU backend not available
 */
template <typename T>
SGVector<T> from_gpu(const SGVector<T>& vec)
{
	if (vec.on_gpu())
	{
		LinalgBackendBase* gpu_backend = sg_linalg->get_gpu_backend();
		if (gpu_backend)
		{
			typedef typename std::aligned_storage<sizeof(T), alignof(T)>::type aligned_t;
			T* data;
			data = reinterpret_cast<T*>(SG_MALLOC(aligned_t, vec.size()));
			gpu_backend->from_gpu(vec, data);
			return SGVector<T>(data, vec.size());
		}
		else
		{
			SG_SERROR("Data memory on GPU but no GPU backend registered. \
						This can happen if the GPU backend was de-activated \
						after memory has been transferred to GPU.\n");
			return false;
		}
	}
	else
	{
		SG_SWARNING("The data is already on CPU.\n");
		return vec;
	}

}

/**
 * Fetches data from GPU memory.
 *
 * @param vector SGMatrix to be transferred
 * @return SGMatrix with matrix on CPU if GPU backend is still available
 * and a shallow-copy of SGMatrix with matrix on GPU if GPU backend not available
 */
template <typename T>
SGMatrix<T> from_gpu(const SGMatrix<T>& mat)
{
	if (mat.on_gpu())
	{
		LinalgBackendBase* gpu_backend = sg_linalg->get_gpu_backend();
		if (gpu_backend)
		{
			typedef typename std::aligned_storage<sizeof(T), alignof(T)>::type aligned_t;
			T* data;
			data = reinterpret_cast<T*>(SG_MALLOC(aligned_t, mat.num_rows*mat.num_cols));
			gpu_backend->from_gpu(mat, data);
			return SGMatrix<T>(data, mat.num_rows, mat.num_cols);
		}
		else
		{
			SG_SERROR("Data memory on GPU but no GPU backend registered. \
						This can happen if the GPU backend was de-activated \
						after memory has been transferred to GPU.\n");
			return false;
		}
	}
	else
	{
		SG_SWARNING("The data is already on CPU.\n");
		return mat;
	}
}

/**
 * Performs the operation result = alpha*a + beta*b.
 *
 * @param a first vector
 * @param b second vector
 * @param result the vector that saves the result
 * @param alpha constant to be multiplied by the first vector
 * @param beta constant to be multiplied by the second vector
 */
template <typename T>
void add(SGVector<T>& a, SGVector<T>& b, SGVector<T>& result, T alpha=1, T beta=1)
{
	REQUIRE(a.vlen == b.vlen, "Length of vector a (%d) doesn't match vector b (%d).\n", a.vlen, b.vlen);
	REQUIRE(result.vlen == b.vlen, "Length of vector result (%d) doesn't match vector a (%d).\n", result.vlen, a.vlen);

	REQUIRE(!(result.on_gpu()^a.on_gpu()),
		"Cannot operate with vector result on_gpu (%d) and vector a on_gpu (%d).\n", result.on_gpu(), a.on_gpu());
	REQUIRE(!(result.on_gpu()^b.on_gpu()),
		"Cannot operate with vector result on_gpu (%d) and vector b on_gpu (%d).\n", result.on_gpu(), b.on_gpu());

	infer_backend(a, b)->add(a, b, alpha, beta, result);
}

/**
 * Performs the operation result = alpha*a + beta*b.
 *
 * @param a first matrix
 * @param b second matrix
 * @param result the matrix that saves the result
 * @param alpha constant to be multiplied by the first matrix
 * @param beta constant to be multiplied by the second matrix
 */
template <typename T>
void add(SGMatrix<T>& a, SGMatrix<T>& b, SGMatrix<T>& result, T alpha=1, T beta=1)
{
	REQUIRE((a.num_rows == b.num_rows), "Number of rows of matrix a (%d) must match matrix b (%d).\n",
		a.num_rows, b.num_rows);
	REQUIRE((a.num_cols == b.num_cols), "Number of columns of matrix a (%d) must match matrix b (%d).\n",
		a.num_cols, b.num_cols);

	REQUIRE(!(result.on_gpu()^a.on_gpu()),
		"Cannot operate with matrix result on_gpu (%d) and matrix a on_gpu (%d).\n", result.on_gpu(), a.on_gpu());
	REQUIRE(!(result.on_gpu()^b.on_gpu()),
		"Cannot operate with matrix result on_gpu (%d) and matrix b on_gpu (%d).\n", result.on_gpu(), b.on_gpu());

	infer_backend(a, b)->add(a, b, alpha, beta, result);
}

/**
 * Performs the operation C = alpha*A + beta*B.
 * @param A first vector or matrix
 * @param B second vector or matrix
 * @param alpha constant to be multiplied by the first vector or matrix
 * @param beta constant to be multiplied by the second vector or matrix
 * @return The result vector or matrix
 */
template <typename T, template<typename> class Container>
Container<T> add(Container<T>& a, Container<T>& b, T alpha=1, T beta=1)
{
	auto result = a.clone();
	add(a, b, result, alpha, beta);
	return result;
}

/**
 * Vector dot-product that works with generic vectors.
 *
 * @param a first vector
 * @param b second vector
 * @return the dot product of \f$\mathbf{a}\f$ and \f$\mathbf{b}\f$, represented
 * as \f$\sum_i a_i b_i\f$
 */
template <typename T>
T dot(const SGVector<T>& a, const SGVector<T>& b)
{
	REQUIRE(a.vlen == b.vlen, "Length of vector a (%d) doesn't match vector b (%d).\n", a.vlen, b.vlen);
	return infer_backend(a, b)->dot(a, b);
}

/** Performs the operation C = A .* B where ".*" denotes elementwise multiplication
 * on matrix blocks.
 *
 * This version returns the result in-place.
 * User should pass an appropriately pre-allocated memory matrix
 *
 * @param a First matrix block
 * @param b Second matrix block
 * @param c Result matrix
 */
template <typename T>
void element_prod(Block<SGMatrix<T>>& a, Block<SGMatrix<T>>& b, SGMatrix<T>& result)
{
	REQUIRE(a.m_row_size == b.m_row_size && a.m_col_size == b.m_col_size,
			"Dimension mismatch! A(%d x %d) vs B(%d x %d)\n",
			a.m_row_size, a.m_col_size, b.m_row_size, b.m_col_size);
	REQUIRE(a.m_row_size == result.num_rows && a.m_col_size == result.num_cols,
			"Dimension mismatch! A(%d x %d) vs result(%d x %d)\n",
			a.m_row_size, a.m_col_size, result.num_rows, result.num_cols);

	REQUIRE(!result.on_gpu(), "Cannot operate with matrix result on_gpu (%d) \
	 		as matrix blocks are on CPU.\n", result.on_gpu());
	sg_linalg->get_cpu_backend()->element_prod(a, b, result);
}

/** Performs the operation C = A .* B where ".*" denotes elementwise multiplication
 * on matrix blocks.
 *
 * This version returns the result in a newly created matrix.
 *
 * @param A First matrix block
 * @param B Second matrix block
 * @return The result of the operation
 */
template <typename T>
SGMatrix<T> element_prod(Block<SGMatrix<T>>& a, Block<SGMatrix<T>>& b)
{
	REQUIRE(a.m_row_size == b.m_row_size && a.m_col_size == b.m_col_size,
			"Dimension mismatch! A(%d x %d) vs B(%d x %d)\n",
			a.m_row_size, a.m_col_size, b.m_row_size, b.m_col_size);

	SGMatrix<T> result(a.m_row_size, a.m_col_size);
	result.zero();

	element_prod(a, b, result);

	return result;
}

/** Performs the operation C = A .* B where ".*" denotes elementwise multiplication.
 *
 * This version returns the result in-place.
 * User should pass an appropriately pre-allocated memory matrix
 * Or pass one of the operands arguments (A or B) as a result
 *
 * @param a First matrix
 * @param b Second matrix
 * @param result Result matrix
 */
template <typename T>
void element_prod(SGMatrix<T>& a, SGMatrix<T>& b, SGMatrix<T>& result)
{
	REQUIRE(a.num_rows == b.num_rows && a.num_cols == b.num_cols,
			"Dimension mismatch! A(%d x %d) vs B(%d x %d)\n",
			a.num_rows, a.num_cols, b.num_rows, b.num_cols);
	REQUIRE(a.num_rows == result.num_rows && a.num_cols == result.num_cols,
			"Dimension mismatch! A(%d x %d) vs result(%d x %d)\n",
			a.num_rows, a.num_cols, result.num_rows, result.num_cols);

	REQUIRE(!(result.on_gpu()^a.on_gpu()),
			"Cannot operate with matrix result on_gpu (%d) and \
			 matrix A on_gpu (%d).\n", result.on_gpu(), a.on_gpu());
	REQUIRE(!(result.on_gpu()^b.on_gpu()),
			"Cannot operate with matrix result on_gpu (%d) and \
			 matrix B on_gpu (%d).\n", result.on_gpu(), b.on_gpu());
	infer_backend(a, b)->element_prod(a, b, result);
}

/** Performs the operation C = A .* B where ".*" denotes elementwise multiplication.
 *
 * This version returns the result in a newly created matrix.
 *
 * @param A First matrix
 * @param B Second matrix
 * @return The result of the operation
 */
template <typename T>
SGMatrix<T> element_prod(SGMatrix<T>& a, SGMatrix<T>& b)
{
	REQUIRE(a.num_rows == b.num_rows && a.num_cols == b.num_cols,
			"Dimension mismatch! A(%d x %d) vs B(%d x %d)\n",
			a.num_rows, a.num_cols, b.num_rows, b.num_cols);

	SGMatrix<T> result;
	result = a.clone();

	element_prod(a, b, result);

	return result;
}

/** Performs the operation of matrix multiply a vector \f$x = Ab\f$.
 *
 * This version returns the result in-place.
 * User should pass an appropriately allocate memory matrix.
 *
 * @param A The matrix
 * @param b The vector
 * @param transpose Whether to transpose the matrix. Default false
 * @param result Result vector
 */
template <typename T>
void matrix_prod(SGMatrix<T>& A, SGVector<T>& b, SGVector<T>& result, bool transpose = false)
{
	if (transpose)
	{
		REQUIRE(A.num_rows == b.vlen, "Row number of Matrix A (%d) doesn't match \
		length of vector b (%d).\n", A.num_rows, b.vlen);
		REQUIRE(result.vlen == A.num_cols, "Length of vector result (%d) doesn't match \
		column number of Matrix A (%d).\n", result.vlen, A.num_cols);
	}
	else
	{
		REQUIRE(A.num_cols == b.vlen, "Column number of Matrix A (%d) doesn't match \
		length of vector b (%d).\n", A.num_cols, b.vlen);
		REQUIRE(result.vlen == A.num_rows, "Length of vector result (%d) doesn't match \
		row number of Matrix A (%d).\n", result.vlen, A.num_rows);
	}

	REQUIRE(!(result.on_gpu()^A.on_gpu()),
		"Cannot operate with vector result on_gpu (%d) and vector a on_gpu (%d).\n", result.on_gpu(), A.on_gpu());
	REQUIRE(!(result.on_gpu()^b.on_gpu()),
		"Cannot operate with vector result on_gpu (%d) and vector b on_gpu (%d).\n", result.on_gpu(), b.on_gpu());

	infer_backend(A, SGMatrix<T>(b))->matrix_prod(A, b, result, transpose, false);
}

/** Performs the operation of matrix multiply a vector \f$x = Ab\f$.
 *
 * @param A The matrix
 * @param b The vector
 * @param transpose Whether to transpose a matrix. Default false
 * @return result Result vector
 */
template <typename T>
SGVector<T> matrix_prod(SGMatrix<T>& A, SGVector<T>& b, bool transpose = false)
{
	SGVector<T> result;
	if (transpose)
	{
		REQUIRE(A.num_rows == b.vlen, "Row number of Matrix A (%d) doesn't match \
		length of vector b (%d).\n", A.num_rows, b.vlen);

		result.resize_vector(A.num_cols);
	}
	else
	{
		REQUIRE(A.num_cols == b.vlen, "Column number of Matrix A (%d) doesn't match \
		length of vector b (%d).\n", A.num_cols, b.vlen);

		result.resize_vector(A.num_rows);
	}

	if (A.on_gpu())
		result = to_gpu(result);

	matrix_prod(A, b, result, transpose);
	return result;
}

/** Performs the operation C = A * B where "*" denotes matrix multiplication.
 *
 * This version returns the result in-place.
 * User should pass an appropriately allocate memory matrix
 *
 * @param A First matrix
 * @param B Second matrix
 * @param result Result matrix
 * @param transpose_A whether to transpose matrix A
 * @param transpose_B whether to transpose matrix B
 */
template <typename T>
void matrix_prod(SGMatrix<T>& A, SGMatrix<T>& B, SGMatrix<T>& result,
	bool transpose_A=false, bool transpose_B=false)
{
	REQUIRE(!(result.on_gpu()^A.on_gpu()),
			"Cannot operate with matrix result on_gpu (%d) and \
			 matrix A on_gpu (%d).\n", result.on_gpu(), A.on_gpu());
	REQUIRE(!(result.on_gpu()^B.on_gpu()),
			"Cannot operate with matrix result on_gpu (%d) and \
			 matrix B on_gpu (%d).\n", result.on_gpu(), B.on_gpu());

	if (transpose_A)
	{
		REQUIRE(A.num_cols == result.num_rows, "Number of columns for A (%d) and "
				"number of rows for result (%d) should be equal!\n", A.num_cols, result.num_rows);
		if (transpose_B)
		{
			REQUIRE(A.num_rows == B.num_cols, "Number of rows for A (%d) and "
					"number of columns for B (%d) should be equal!\n", A.num_rows, B.num_cols);
			REQUIRE(B.num_rows == result.num_cols, "Number of rows for B (%d) and "
					"number of columns for result (%d) should be equal!\n",
					B.num_rows, result.num_cols);
		}
		else
		{
			REQUIRE(A.num_rows == B.num_rows, "Number of rows for A (%d) and "
					"number of rows for B (%d) should be equal!\n", A.num_rows, B.num_rows);
			REQUIRE(B.num_cols == result.num_cols, "Number of columns for B (%d) and "
					"number of columns for result (%d) should be equal!\n",
					B.num_cols, result.num_cols);
		}
	}
	else
	{
		REQUIRE(A.num_rows == result.num_rows, "Number of rows for A (%d) and "
				"number of rows for result (%d) should be equal!\n", A.num_rows, result.num_rows);
		if (transpose_B)
		{
			REQUIRE(A.num_cols == B.num_cols, "Number of columns for A (%d) and "
					"number of columns for B (%d) should be equal!\n", A.num_cols, B.num_cols);
			REQUIRE(B.num_rows == result.num_cols, "Number of rows for B (%d) and "
					"number of columns for result (%d) should be equal!\n",
					B.num_rows, result.num_cols);
		}
		else
		{
			REQUIRE(A.num_cols == B.num_rows, "Number of columns for A (%d) and "
					"number of rows for B (%d) should be equal!\n", A.num_cols, B.num_rows);
			REQUIRE(B.num_cols == result.num_cols, "Number of columns for B (%d) and "
					"number of columns for result (%d) should be equal!\n",
					B.num_cols, result.num_cols);
		}
	}

	infer_backend(A, B)->matrix_prod(A, B, result, transpose_A, transpose_B);
}

/** Performs the operation C = A * B where "*" denotes matrix multiplication.
 *
 * This version returns the result in a newly created matrix.
 *
 * @param A First matrix
 * @param B Second matrix
 * @param transpose_A whether to transpose matrix A
 * @param transpose_B whether to transpose matrix B
 *
 * @return The result of the operation
 */
template <typename T>
SGMatrix<T> matrix_prod(SGMatrix<T>& A, SGMatrix<T>& B,
	bool transpose_A=false, bool transpose_B=false)
{
	SGMatrix<T> result;

	if (transpose_A & transpose_B)
	{
		REQUIRE(A.num_rows == B.num_cols, "Number of rows for A (%d) and "
				"number of columns for B (%d) should be equal!\n", A.num_rows, B.num_cols);
		result = SGMatrix<T>(A.num_cols, B.num_rows);
	}
	else if (transpose_A)
	{
		REQUIRE(A.num_rows == B.num_rows, "Number of rows for A (%d) and "
				"number of rows for B (%d) should be equal!\n", A.num_rows, B.num_rows);
		result = SGMatrix<T>(A.num_cols, B.num_cols);
	}
	else if (transpose_B)
	{
		REQUIRE(A.num_cols == B.num_cols, "Number of columns for A (%d) and "
				"number of columns for B (%d) should be equal!\n", A.num_cols, B.num_cols);
		result = SGMatrix<T>(A.num_rows, B.num_rows);
	}
	else
	{
		REQUIRE(A.num_cols == B.num_rows, "Number of columns for A (%d) and "
				"number of rows for B (%d) should be equal!\n", A.num_cols, B.num_rows);
		result = SGMatrix<T>(A.num_rows, B.num_cols);
	}

	if (A.on_gpu())
		result = to_gpu(result);

	matrix_prod(A, B, result, transpose_A, transpose_B);

	return result;
}

/**
 * Returns the largest element in a vector or matrix
 *
 * @param a input vector or matrix
 * @return largest value in the vector or matrix
 */
template <typename T, template <typename> class Container>
T max(const Container<T>& a)
{
	return infer_backend(a)->max(a);
}

/**
 * Method that computes the mean of vectors or matrices composed of real numbers.
 *
 * @param a SGVector or SGMatrix
 * @return the vector mean \f$\bar{a}_i\f$ or matrix mean \f$\bar{m}_{i,j}\f$
 */
template<typename T, template<typename> class Container>
typename std::enable_if<!std::is_same<T, complex128_t>::value, float64_t>::type
mean(const Container<T>& a)
{
	REQUIRE(a.size() > 0, "Vector/Matrix cannot be empty!\n");
	return infer_backend(a)->mean(a);
}

/**
 * Method that computes the mean of vectors or matrices composed of complex numbers.
 *
 * @param a SGVector or SGMatrix
 * @return the vector mean \f$\bar{a}_i\f$ or matrix mean \f$\bar{m}_{i,j}\f$
 */
template<template<typename> class Container>
complex128_t mean(const Container<complex128_t>& a)
{
	REQUIRE(a.size() > 0, "Vector/Matrix cannot be empty!\n");
	return infer_backend(a)->mean(a);
}

/**
 * Range fill a vector or matrix with start...start+len-1
 * @param a the vector or matrix to be filled
 * @param start value to be assigned to first element of vector or matrix
 */
template <typename T, template<typename> class Container>
void range_fill(Container<T>& a, const T start=0)
{
	infer_backend(a)->range_fill(a, start);
}

/**
 * Performs the operation result = alpha * A on vectors
 *
 * @param a first vector
 * @param alpha scale factor
 * @param result the vector of alpha * A
 */
template <typename T>
void scale(SGVector<T>& a, SGVector<T>& result, T alpha=1)
{
	REQUIRE(result.vlen == a.vlen, "Length of vector result (%d) doesn't match vector a (%d).\n", result.vlen, a.vlen);
	infer_backend(a, result)->scale(a, alpha, result);
}

/**
 * Performs the operation result = alpha * A on matrices
 *
 * @param a first matrix
 * @param alpha scale factor
 * @param result the matrix of alpha * A
 */
template <typename T>
void scale(SGMatrix<T>& a, SGMatrix<T>& result, T alpha=1)
{
	REQUIRE((a.num_rows == result.num_rows), "Number of rows of matrix a (%d) must match matrix result (%d).\n",
		a.num_rows, result.num_rows);
	REQUIRE((a.num_cols == result.num_cols), "Number of columns of matrix a (%d) must match matrix result (%d).\n",
		a.num_cols, result.num_cols);
	infer_backend(a, result)->scale(a, alpha, result);
}

/**
 * Performs the operation B = alpha * A on vectors or matrices
 *
 * @param a first vector/matrix
 * @param alpha scale factor
 * @return vector or matrix of alpha * A
 */
template<typename T, template<typename> class Container>
Container<T> scale(Container<T>& a, T alpha=1)
{
	auto result = a.clone();
	scale(a, result, alpha);
	return result;
}

/**
 * Set const value to vectors or matrices
 *
 * @param a vector or matrix to be set
 * @param value the value to set the vector or matrix
 */
template <typename T, template<typename> class Container>
void set_const(Container<T>& a, T value)
{
	infer_backend(a)->set_const(a, value);
}

/** Sets each row of a matrix to some constant value. That is, perfoms the
 * operation A[i,j] = v[i], for all i and j
 *
 * @param a matrix to be set
 * @param the vector to set the matrix
 */
template <typename T>
void set_rows_const(SGMatrix<T>& a, const SGVector<T>& b)
{
	infer_backend(a, SGMatrix<T>(b))->set_rows_const(a, b);
}

/**
 * Method that computes the sum of vectors or matrices
 *
 * @param a the vector or matrix whose sum has to be computed
 * @param no_diag if true, diagonal entries are excluded from the sum
 * @return the vector sum \f$\sum_i a_i\f$ or matrix sum \f$\sum_{i,j}b_{i,j}\f$
 */
template <typename T, template <typename> class Container>
T sum(const Container<T>& a, bool no_diag=false)
{
	return infer_backend(a)->sum(a, no_diag);
}

/**
 * Method that computes the sum of matrix blocks
 *
 * @param a the matrix-block whose sum of co-efficients has to be computed
 * @param no_diag if true, diagonal entries are excluded from the sum
 * @return matrix-block sum \f$\sum_{i,j}b_{i,j}\f$
 */
template <typename T>
T sum(const Block<SGMatrix<T>>& a, bool no_diag=false)
{
	return sg_linalg->get_cpu_backend()->sum(a, no_diag);
}

/**
 * Method that computes the sum of symmetric matrices
 *
 * @param a the symmetric matrix whose sum has to be computed
 * @param no_diag if true, diagonal entries are excluded from the sum
 * @return the matrix sum \f$\sum_{i,j}b_{i,j}\f$
 */
template <typename T, template <typename> class SGMatrix>
T sum_symmetric(const SGMatrix<T>& a, bool no_diag=false)
{
	REQUIRE(a.num_rows == a.num_cols, "Matrix is not square!\n");
	return infer_backend(a)->sum_symmetric(a, no_diag);
}

/**
 * Method that computes the sum of symmetric matrix blocks
 *
 * @param a the symmetric matrix-block whose sum has to be computed
 * @param no_diag if true, diagonal entries are excluded from the sum
 * @return symmetric matrix-block sum \f$\sum_{i,j}b_{i,j}\f$
 */
template <typename T>
T sum_symmetric(const Block<SGMatrix<T>>& a, bool no_diag=false)
{
	REQUIRE(a.m_row_size == a.m_col_size, "Matrix is not square!\n");
	return sg_linalg->get_cpu_backend()->sum_symmetric(a, no_diag);
}

/**
 * Method that computes colwise sum of co-efficients of a dense matrix
 *
 * @param mat a matrix whose colwise sum has to be computed
 * @return the colwise sum of co-efficients computed as \f$s_j=\sum_{i}b_{i,j}\f$
 */
template <typename T>
SGVector<T> colwise_sum(const SGMatrix<T>& mat, bool no_diag=false)
{
	return infer_backend(mat)->colwise_sum(mat, no_diag);
}

/**
 * Method that computes the colwise sum of matrix blocks
 *
 * @param a the matrix-block whose colwise sum of co-efficients has to be computed
 * @param no_diag if true, diagonal entries are excluded from the sum
 * @return the colwise sum of co-efficients computed as \f$s_j=\sum_{i}b_{i,j}\f$
 */
template <typename T>
SGVector<T> colwise_sum(const Block<SGMatrix<T>>& a, bool no_diag=false)
{
	return sg_linalg->get_cpu_backend()->colwise_sum(a, no_diag);
}

/**
 * Method that computes rowwise sum of co-efficients of a dense matrix
 *
 * @param mat a matrix whose rowwise sum has to be computed
 * @return the rowwise sum of co-efficients computed as \f$s_i=\sum_{j}m_{i,j}\f$
 */
template <typename T>
SGVector<T> rowwise_sum(const SGMatrix<T>& mat, bool no_diag=false)
{
	return infer_backend(mat)->rowwise_sum(mat, no_diag);
}

/**
 * Method that computes the rowwise sum of matrix blocks
 *
 * @param a the matrix-block whose rowwise sum of co-efficients has to be computed
 * @param no_diag if true, diagonal entries are excluded from the sum
 * @return the rowwise sum of co-efficients computed as \f$s_i=\sum_{j}m_{i,j}\f$
 */
template <typename T>
SGVector<T> rowwise_sum(const Block<SGMatrix<T>>& a, bool no_diag=false)
{
	return sg_linalg->get_cpu_backend()->rowwise_sum(a, no_diag);
}

}

}
