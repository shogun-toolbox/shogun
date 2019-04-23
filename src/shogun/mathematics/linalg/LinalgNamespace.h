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

#ifndef LINALG_NAMESPACE_H_
#define LINALG_NAMESPACE_H_

#include <shogun/base/ShogunEnv.h>
#include <shogun/mathematics/linalg/LinalgBackendBase.h>
#include <shogun/mathematics/linalg/LinalgEnums.h>
#include <shogun/mathematics/linalg/SGLinalg.h>

namespace shogun
{

	namespace linalg
	{

		/** Infer the appropriate backend for linalg operations
		 * from the input SGVector or SGMatrix (Container).
		 *
		 * @param a SGVector or SGMatrix
		 * @return @see LinalgBackendBase pointer
		 */
		template <typename T, template <typename> class Container>
		LinalgBackendBase* infer_backend(const Container<T>& a)
		{
			auto sg_linalg = env()->linalg();
			if (a.on_gpu())
			{
				if (sg_linalg->get_gpu_backend())
					return sg_linalg->get_gpu_backend();
				else
				{
					error(
					    "Vector or matrix is on GPU but no GPU backend registered. \
						This can happen if the GPU backend was de-activated \
						after memory has been transferred to GPU.");
					return NULL;
				}
			}
			else
				return sg_linalg->get_cpu_backend();
		}

		/** Infer the appropriate backend for linalg operations
		 * from the input SGVector or SGMatrix (Container).
		 * Raise error if the backends of the two Containers conflict.
		 *
		 * @param a The first SGVector/SGMatrix
		 * @param b The second SGVector/SGMatrix
		 * @return @see LinalgBackendBase pointer
		 */
		template <typename T, typename U, template <typename> class Container>
		LinalgBackendBase*
		infer_backend(const Container<T>& a, const Container<U>& b)
		{
			auto sg_linalg = env()->linalg();
			if (a.on_gpu() && b.on_gpu())
			{
				if (sg_linalg->get_gpu_backend())
					return sg_linalg->get_gpu_backend();
				else
				{
					error(
					    "Vector or matrix is on GPU but no GPU backend registered. \
					  This can happen if the GPU backend was de-activated \
					  after memory has been transferred to GPU.");
					return NULL;
				}
			}
			else if (a.on_gpu() || b.on_gpu())
			{
				error(
				    "Cannot operate with first vector/matrix on_gpu flag({}) \
					and second vector/matrix on_gpu flag ({}).",
				    a.on_gpu(), b.on_gpu());
				return NULL;
			}
			else
				return sg_linalg->get_cpu_backend();
		}

		/** Infer the appropriate backend for linalg operations
		 * from the input SGVector or SGMatrix (Container).
		 * Raise error if the backends of the Containers conflict.
		 *
		 * @param a The first SGVector/SGMatrix
		 * @param b The second SGVector/SGMatrix
		 * @param c The third SGVector/SGMatrix
		 * @return @see LinalgBackendBase pointer
		 */
		template <typename T, template <typename> class Container>
		LinalgBackendBase* infer_backend(
		    const Container<T>& a, const Container<T>& b, const Container<T>& c)
		{
			auto sg_linalg = env()->linalg();
			if (a.on_gpu() && b.on_gpu() && c.on_gpu())
			{
				if (sg_linalg->get_gpu_backend())
					return sg_linalg->get_gpu_backend();
				else
				{
					error(
					    "Vector or matrix is on GPU but no GPU backend registered. \
					  This can happen if the GPU backend was de-activated \
					  after memory has been transferred to GPU.");
					return NULL;
				}
			}
			else if (a.on_gpu() || b.on_gpu() || c.on_gpu())
			{
				error(
				    "Cannot operate with first vector/matrix on_gpu flag({}),\
					second vector/matrix on_gpu flag ({}) and third vector/matrix \
					on_gpu flag ({}).",
				    a.on_gpu(), b.on_gpu(), c.on_gpu());
				return NULL;
			}
			else
				return sg_linalg->get_cpu_backend();
		}

		/**
		 * Transfers data to GPU memory.
		 * Shallow-copy of SGVector with vector on CPU if GPU backend not
		 * available
		 *
		 * @param a SGVector to be transferred
		 * @param b SGVector to be set
		 */
		template <typename T>
		void to_gpu(const SGVector<T>& a, SGVector<T>& b)
		{
			auto sg_linalg = env()->linalg();
			sg_linalg->m_gpu_transfer.lock();

			if (a.on_gpu())
			{
				if (sg_linalg->get_linalg_warnings())
					io::warn("The vector is already on GPU.");
			}
			else
			{
				LinalgBackendBase* gpu_backend = sg_linalg->get_gpu_backend();

				if (gpu_backend)
					b = SGVector<T>(gpu_backend->to_gpu(a), a.vlen);
				else
				{
					if (sg_linalg->get_linalg_warnings())
						io::warn("Trying to access GPU memory\
				 	without GPU backend registered.");
					b = a;
				}
			}

			sg_linalg->m_gpu_transfer.unlock();
		}

		/**
		 * Transfers data to GPU memory. Does nothing if no GPU backend
		 * registered.
		 * Shallow-copy SGMatrix on CPU if GPU backend not available
		 *
		 * @param a SGMatrix to be transferred
		 * @param b SGMatrix to be set
		 */
		template <typename T>
		void to_gpu(const SGMatrix<T>& a, SGMatrix<T>& b)
		{
			auto sg_linalg = env()->linalg();
			sg_linalg->m_gpu_transfer.lock();

			if (a.on_gpu())
			{
				if (sg_linalg->get_linalg_warnings())
					io::warn("The matrix is already on GPU.");
			}
			else
			{
				LinalgBackendBase* gpu_backend = sg_linalg->get_gpu_backend();

				if (gpu_backend)
					b = SGMatrix<T>(
					    gpu_backend->to_gpu(a), a.num_rows, a.num_cols);
				else
				{
					if (sg_linalg->get_linalg_warnings())
						io::warn("Trying to access GPU memory\
				 	without GPU backend registered.");
					b = a;
				}
			}

			sg_linalg->m_gpu_transfer.unlock();
		}

		/**
		* Transfers data to GPU memory in-place.
		*
		* @param a SGVector or SGMatrix to be transferred
		*/
		template <typename T, template <typename> class Container>
		void to_gpu(Container<T>& a)
		{
			to_gpu(a, a);
		}

		/**
		 * Fetches data from GPU memory.
		 * Transfer vectors to CPU if GPU backend is still available.
		 *
		 * @param a SGVector to be transferred
		 * @param b SGVector to be set
		 */
		template <typename T>
		void from_gpu(SGVector<T>& a, SGVector<T>& b)
		{
			auto sg_linalg = env()->linalg();
			sg_linalg->m_gpu_transfer.lock();
			if (a.on_gpu())
			{
				LinalgBackendBase* gpu_backend = sg_linalg->get_gpu_backend();
				if (gpu_backend)
				{
					typedef typename std::aligned_storage<
					    sizeof(T), alignof(T)>::type aligned_t;
					T* data;
					data = reinterpret_cast<T*>(SG_MALLOC(aligned_t, a.size()));
					gpu_backend->from_gpu(a, data);
					b = SGVector<T>(data, a.size());
				}
				else
					error(
					    "Data memory on GPU but no GPU backend registered. \
						This can happen if the GPU backend was de-activated \
						after memory has been transferred to GPU.");
			}
			else
			{
				if (sg_linalg->get_linalg_warnings())
					io::warn("The data is already on CPU.");
				b = a;
			}

			sg_linalg->m_gpu_transfer.unlock();
		}

		/**
		 * Fetches data from GPU memory.
		 * Transfer matrices to CPU if GPU backend is still available.
		 *
		 * @param a SGMatrix to be transferred
		 * @param b SGMatrix to be set
		 */
		template <typename T>
		void from_gpu(SGMatrix<T>& a, SGMatrix<T>& b)
		{
			auto sg_linalg = env()->linalg();
			sg_linalg->m_gpu_transfer.lock();
			if (a.on_gpu())
			{
				LinalgBackendBase* gpu_backend = sg_linalg->get_gpu_backend();
				if (gpu_backend)
				{
					typedef typename std::aligned_storage<
					    sizeof(T), alignof(T)>::type aligned_t;
					T* data;
					data = reinterpret_cast<T*>(
					    SG_MALLOC(aligned_t, a.num_rows * a.num_cols));
					gpu_backend->from_gpu(a, data);
					b = SGMatrix<T>(data, a.num_rows, a.num_cols);
				}
				else
					error(
					    "Data memory on GPU but no GPU backend registered. \
						This can happen if the GPU backend was de-activated \
						after memory has been transferred to GPU.");
			}
			else
			{
				if (sg_linalg->get_linalg_warnings())
					io::warn("The data is already on CPU.");
				b = a;
			}

			sg_linalg->m_gpu_transfer.unlock();
		}

		/**
		 * Fetches data from GPU memory.
		 * Transfer vector or matrix to CPU if GPU backend is still available.
		 *
		 * @param a SGVector or SGMatrix to be transferred
		*/
		template <typename T, template <typename> class Container>
		void from_gpu(Container<T>& a)
		{
			from_gpu(a, a);
		}

		/**
		 * Performs the operation result = alpha * a + beta * b on vectors.
		 * This version returns the result in-place.
		 * User should pass an appropriately pre-allocated memory matrix
		 * Or pass one of the operands arguments (A or B) as a result
		 *
		 * @param a First vector
		 * @param b Second vector
		 * @param result The vector that saves the result
		 * @param alpha Constant to be multiplied by the first vector
		 * @param beta Constant to be multiplied by the second vector
		 */
		template <typename T>
		void
		add(const SGVector<T>& a, const SGVector<T>& b, SGVector<T>& result,
		    T alpha = 1, T beta = 1)
		{
			require(
			    a.vlen == b.vlen,
			    "Length of vector a ({}) doesn't match vector b ({}).",
			    a.vlen, b.vlen);
			require(
			    result.vlen == b.vlen,
			    "Length of vector result ({}) doesn't match vector a ({}).",
			    result.vlen, a.vlen);

			require(
			    !(result.on_gpu() ^ a.on_gpu()), "Cannot operate with vector "
			                                     "result on_gpu ({}) and "
			                                     "vector a on_gpu ({}).",
			    result.on_gpu(), a.on_gpu());
			require(
			    !(result.on_gpu() ^ b.on_gpu()), "Cannot operate with vector "
			                                     "result on_gpu ({}) and "
			                                     "vector b on_gpu ({}).",
			    result.on_gpu(), b.on_gpu());

			infer_backend(a, b)->add(a, b, alpha, beta, result);
		}

		/**
		 * Performs the operation result = alpha * a + beta * b on matrices.
		 * This version returns the result in-place.
		 * User should pass an appropriately pre-allocated memory matrix
		 * Or pass one of the operands arguments (A or B) as a result
		 *
		 * @param a First matrix
		 * @param b Second matrix
		 * @param result The matrix that saves the result
		 * @param alpha Constant to be multiplied by the first matrix
		 * @param beta Constant to be multiplied by the second matrix
		 */
		template <typename T>
		void
		add(const SGMatrix<T>& a, const SGMatrix<T>& b, SGMatrix<T>& result,
		    T alpha = 1, T beta = 1)
		{
			require(
			    (a.num_rows == b.num_rows),
			    "Number of rows of matrix a ({}) must match matrix b ({}).",
			    a.num_rows, b.num_rows);
			require(
			    (a.num_cols == b.num_cols), "Number of columns of matrix a "
			                                "({}) must match matrix b ({}).",
			    a.num_cols, b.num_cols);

			require(
			    !(result.on_gpu() ^ a.on_gpu()), "Cannot operate with matrix "
			                                     "result on_gpu ({}) and "
			                                     "matrix a on_gpu ({}).",
			    result.on_gpu(), a.on_gpu());
			require(
			    !(result.on_gpu() ^ b.on_gpu()), "Cannot operate with matrix "
			                                     "result on_gpu ({}) and "
			                                     "matrix b on_gpu ({}).",
			    result.on_gpu(), b.on_gpu());

			infer_backend(a, b)->add(a, b, alpha, beta, result);
		}

		/**
		 * Performs the operation C = alpha * A + beta * B.
		 * This version returns the result in a newly created vector or matrix.
		 *
		 * @param a First vector or matrix
		 * @param b Second vector or matrix
		 * @param alpha Constant to be multiplied by the first vector or matrix
		 * @param beta Constant to be multiplied by the second vector or matrix
		 * @return The result vector or matrix
		 */
		template <typename T, template <typename> class Container>
		Container<T>
		add(const Container<T>& a, const Container<T>& b, T alpha = 1,
		    T beta = 1)
		{
			auto result = a.clone();
			add(a, b, result, alpha, beta);
			return result;
		}

		/**
		 * Performs the operation result.col(i) = alpha * A.col(i) + beta * b.
		 * User should pass an appropriately pre-allocated memory matrix
		 * Or pass the operand argument A as a result.
		 *
		 * @param A The matrix
		 * @param b The vector
		 * @param result The matrix that saves the result
		 * @param alpha Constant to be multiplied by the matrix
		 * @param beta Constant to be multiplied by the vector
		 */
		template <typename T>
		void add_col_vec(
		    const SGMatrix<T>& A, index_t i, const SGVector<T>& b,
		    SGMatrix<T>& result, T alpha = 1, T beta = 1)
		{
			require(
			    A.num_rows == b.vlen, "Number of rows of matrix A ({}) doesn't "
			                          "match length of vector b ({}).",
			    A.num_rows, b.vlen);
			require(
			    result.num_rows == A.num_rows,
			    "Number of rows of result ({}) doesn't match matrix A ({}).",
			    result.num_rows, A.num_rows);
			require(
			    i >= 0 && i < A.num_cols, "Index i ({}) is out of range (0-{})",
			    i, A.num_cols - 1);

			infer_backend(A, SGMatrix<T>(b))
			    ->add_col_vec(A, i, b, result, alpha, beta);
		}

		/**
		 * Performs the operation result = alpha * A.col(i) + beta * b.
		 * User should pass an appropriately pre-allocated vector
		 * Or pass the operand argument b as a result.
		 *
		 * @param A The matrix
		 * @param b The vector
		 * @param result The vector that saves the result
		 * @param alpha Constant to be multiplied by the matrix
		 * @param beta Constant to be multiplied by the vector
		 */
		template <typename T>
		void add_col_vec(
		    const SGMatrix<T>& A, index_t i, const SGVector<T>& b,
		    SGVector<T>& result, T alpha = 1, T beta = 1)
		{
			require(
			    A.num_rows == b.vlen, "Number of rows of matrix A ({}) doesn't "
			                          "match length of vector b ({}).",
			    A.num_rows, b.vlen);
			require(
			    result.vlen == b.vlen,
			    "Length of result ({}) doesn't match vector b ({}).",
			    result.vlen, b.vlen);
			require(
			    i >= 0 && i < A.num_cols, "Index i ({}) is out of range (0-{})",
			    i, A.num_cols - 1);

			infer_backend(A, SGMatrix<T>(b))
			    ->add_col_vec(A, i, b, result, alpha, beta);
		}

		/** Calculates pseudo inverse A+ from eigen values solved by a self
		 * adjoint eigen solver.
		 * User should pass an appropriately pre-allocated memory matrix
		 * or pass the operand argument A as a result.
		 *
		 * @param A The matrix
		 * @param result The matrix that saves the result
		 */
		template <typename T>
		void pinvh(const SGMatrix<T>& A, SGMatrix<T>& result)
		{

			require(
			    result.num_rows == A.num_rows && result.num_cols == A.num_cols,
			    "Dimension mismatch! A ({} x {}) vs result ({} x {}).",
			    A.num_rows, A.num_cols, result.num_rows, result.num_cols);

			require(
			    A.num_rows == A.num_cols,
			    "Given matrix is not square ({} x {})", A.num_rows, A.num_cols);

			infer_backend(A)->pinvh(A, result);
		}

		/**
		 * Calculates pseudo inverse A+ from eigen values solved by a self
		 * adjoint eigen solver.
		 * This version returns the result in a newly created matrix.
		 *
		 * @param A The matrix
		 * @return The result matrix
		 */
		template <typename T>
		SGMatrix<T> pinvh(const SGMatrix<T>& A)
		{
			SGMatrix<T> result(A.num_rows, A.num_rows);
			pinvh(A, result);
			return result;
		}

		/** Calculates pseudo inverse A+ using singular value decomposition.
		 * User should pass an appropriately pre-allocated memory matrix
		 *
		 * @param A The matrix
		 * @param result The matrix that saves the result
		 */
		template <typename T>
		void pinv(const SGMatrix<T>& A, SGMatrix<T>& result)
		{
			require(
			    result.num_rows == A.num_cols && result.num_cols == A.num_rows,
			    "Dimension mismatch! Result must be of ({} x {}) provided is "
			    "({} x {}).",
			    A.num_cols, A.num_rows, result.num_rows, result.num_cols);

			infer_backend(A)->pinv(A, result);
		}

		/**
		 * Calculates pseudo inverse A+ using singular value decomposition.
		 * This version returns the result in a newly created matrix.
		 *
		 * @param A The matrix
		 * @return The result matrix
		 */
		template <typename T>
		SGMatrix<T> pinv(const SGMatrix<T>& A)
		{
			SGMatrix<T> result(A.num_cols, A.num_rows);
			pinv(A, result);
			return result;
		}

		/**
		 * Performs the operation A.diagonal = alpha * A.diagonal + beta * b.
		 * The matrix is not required to be square.
		 *
		 * @param A The matrix
		 * @param b The vector
		 * @param alpha Constant to be multiplied by the main diagonal of the
		 * matrix
		 * @param beta Constant to be multiplied by the vector
		 */
		template <typename T>
		void
		add_diag(SGMatrix<T>& A, const SGVector<T>& b, T alpha = 1, T beta = 1)
		{
			auto diagonal_len = Math::min(A.num_cols, A.num_rows);
			require(
			    diagonal_len == b.vlen, "Length of main diagonal of matrix A "
			                            "({}) doesn't match length of vector b "
			                            "({}).",
			    diagonal_len, b.vlen);
			require(
			    diagonal_len > 0 && b.vlen > 0, "Matrix / vector can't be "
			                                    "empty.");
			infer_backend(A, SGMatrix<T>(b))->add_diag(A, b, alpha, beta);
		}

		/**
		 * Performs the operation A.diagonal = A.diagonal + beta.
		 * The matrix is not required to be square.
		 *
		 * @param A The matrix
		 * @param beta Constant to be multiplied by the vector
		 */
		template <typename T>
		void add_ridge(SGMatrix<T>& A, T beta)
		{
			auto diagonal_len = Math::min(A.num_cols, A.num_rows);
			require(diagonal_len > 0, "Matrix can't be empty.");
			infer_backend(A)->add_ridge(A, beta);
		}

		/**
		 * Performs the operation result = alpha * A.col(i) + beta * b,
		 * for each column of A.
		 * User should pass an appropriately pre-allocated memory matrix
		 * or pass the operand argument A as a result.
		 *
		 * @param A The matrix
		 * @param b The vector
		 * @param result The matrix that saves the result
		 * @param alpha Constant to be multiplied by the matrix
		 * @param beta Constant to be multiplied by the vector
		 */
		template <typename T>
		void add_vector(
		    const SGMatrix<T>& A, const SGVector<T>& b, SGMatrix<T>& result,
		    T alpha = 1, T beta = 1)
		{
			require(
			    A.num_rows == b.vlen, "Number of rows of matrix A ({}) doesn't "
			                          "match length of vector b ({}).",
			    A.num_rows, b.vlen);
			require(
			    result.num_rows == A.num_rows && result.num_cols == A.num_cols,
			    "Dimension mismatch! A ({} x {}) vs result ({} x {}).",
			    A.num_rows, A.num_cols, result.num_rows, result.num_cols);

			infer_backend(A, SGMatrix<T>(b))
			    ->add_vector(A, b, result, alpha, beta);
		}

		/**
		 * Adds a scalar to the matrix/vector in place.
		 *
		 * @param a Vector or matrix
		 * @param b Scalar to be added
		 */
		template <typename T, template <typename> class Container>
		void add_scalar(Container<T>& a, T b)
		{
			infer_backend(a)->add_scalar(a, b);
		}

		/**
		 * Centers a square matrix in-place, i.e. removes column/row mean from
		 * columns/rows.
		 * In particular it computes A = A - 1N*A - A*1N + 1N*A*1N
		 * where 1N denotes the matrix of the same size as A for which each
		 * element
		 * takes value 1/n, where n is the number of columns and rows of A.
		 *
		 * @param A The matrix to be centered
		 */
		template <typename T>
		void center_matrix(SGMatrix<T>& A)
		{
			require(
			    A.num_rows == A.num_cols, "Matrix A ({} x% d) is not square!",
			    A.num_rows, A.num_cols);
			infer_backend(A)->center_matrix(A);
		}

		/**
		 * Compute the cholesky decomposition \f$A = L L^{*}\f$ or \f$A = U^{*}
		 * U\f$
		 * of a Hermitian positive definite matrix
		 *
		 * @param A The matrix whose Cholesky decomposition is to be computed
		 * @param lower Whether to compute the upper or lower triangular
		 *  Cholesky factorization (default: lower)
		 * @return The upper or lower triangular Cholesky factorization
		 */
		template <typename T>
		SGMatrix<T>
		cholesky_factor(const SGMatrix<T>& A, const bool lower = true)
		{
			require(
			    A.num_rows == A.num_cols,
			    "Matrix dimensions ({}x{}) are not square", A.num_rows,
			    A.num_cols);
			return infer_backend(A)->cholesky_factor(A, lower);
		}

		/**
		 * Updates the Cholesky factorization \f$A = L L^{*}\f$ with a rank one
		 * term in-place.
		 * If A = LL^T before the update, then after it
		 * LL^{*} = A + alpha * b b^{*}
		 *
		 * @param L Triangular matrix, Cholesky factorization of A
		 * @param b Vector whose outer product with itself is the update
		 * @param alpha Scaling factor
		 * @param lower Whether to use L as the upper or lower triangular
		 *  Cholesky factorization (default:lower)
		 */
		template <typename T>
		void cholesky_rank_update(
		    SGMatrix<T>& L, const SGVector<T>& b, T alpha = 1,
		    bool lower = true)
		{
			require(
			    L.num_rows == L.num_cols, "Matrix ({}x{}) is not square",
			    L.num_rows, L.num_cols);
			require(
			    L.num_rows == b.size(),
			    "Vector size ({}) must match matrix size ({}x{})", b.size(),
			    L.num_rows, L.num_rows);
			return infer_backend(L, SGMatrix<T>(b))
			    ->cholesky_rank_update(L, b, alpha, lower);
		}

		/**
		 * Updates a matrix \f$A\f$ with a rank one term in-place,If A = LL^T
		 * before the update, then after it
		 * A = A + alpha * b b^{*}
		 *
		 * @param A square matrix
		 * @param b Vector whose outer product with itself is the update
		 * @param alpha Scaling factor
		 */
		template <typename T>
		void rank_update(SGMatrix<T>& A, const SGVector<T>& b, T alpha = 1)
		{
			require(
			    A.num_rows == A.num_cols, "Matrix ({}x{}) is not square",
			    A.num_rows, A.num_cols);
			require(
			    A.num_rows == b.size(),
			    "Vector size ({}) must match matrix size ({}x{})", b.size(),
			    A.num_rows, A.num_rows);
			return infer_backend(A, SGMatrix<T>(b))->rank_update(A, b, alpha);
		}

		/**
		 * Solve the linear equations \f$Ax=b\f$, given the Cholesky
		 * factorization of A,
		 * where \f$A\f$ is a Hermitian positive definite matrix
		 *
		 * @param L Triangular matrix, Cholesky factorization of A
		 * @param b Right-hand side array
		 * @param lower Whether to use L as the upper or lower triangular
		 *  Cholesky factorization (default:lower)
		 * @return \f$\x\f$
		 */
		template <typename T>
		SGVector<T> cholesky_solver(
		    const SGMatrix<T>& L, const SGVector<T>& b, const bool lower = true)
		{
			require(
			    L.num_rows == L.num_cols,
			    "Matrix dimensions ({}x{}) are not square", L.num_rows,
			    L.num_cols);
			require(
			    L.num_rows == b.size(),
			    "Vector size ({}) must match matrix size ({}x{})", b.size(),
			    L.num_rows);
			return infer_backend(L, SGMatrix<T>(b))
			    ->cholesky_solver(L, b, lower);
		}

		/**
		 * Compute the LDLT cholesky decomposition \f$A = P^{T} L D L^{*} P\f$
		 * or \f$A = P^{T} U^{*} D U P\f$
		 * of a positive semidefinite or negative semidefinite Hermitan matrix
		 *
		 * @param A The matrix whose LDLT cholesky decomposition is to be
		 *  computed
		 * @param L The matrix that saves the triangular LDLT
		 *  Cholesky factorization (default: lower)
		 * @param d The vector that saves the diagonal of the diagonal matrix D
		 * @param p The vector that saves the permutation matrix P as a
		 * transposition sequence
		 * @param lower Whether to use L as the upper or lower triangular
		 *  Cholesky factorization (default:lower)
		 */
		template <typename T>
		void ldlt_factor(
		    const SGMatrix<T>& A, SGMatrix<T>& L, SGVector<T>& d,
		    SGVector<index_t>& p, const bool lower = true)
		{
			require(
			    A.num_rows == A.num_cols,
			    "Matrix dimensions ({}x{}) are not square", A.num_rows,
			    A.num_cols);
			require(
			    A.num_rows == L.num_rows && A.num_cols == L.num_cols,
			    "Shape of matrix A ({}, {}) doesn't match matrix L ({}, {})",
			    A.num_rows, A.num_cols, L.num_rows, L.num_rows);
			require(
			    d.vlen == A.num_rows, "Length of vector d ({}) doesn't match "
			                          "length of diagonal of matrix L ({})",
			    d.vlen, A.num_rows);
			require(
			    p.vlen = A.num_rows, "Length of transpositions vector p ({}) "
			                         "doesn't match length of diagonal of "
			                         "matrix L ({})",
			    p.vlen, A.num_rows);

			infer_backend(A)->ldlt_factor(A, L, d, p, lower);
		}

		/**
		 * Solve the linear equations \f$Ax=b\f$, given the LDLT Cholesky
		 * factorization of A,
		 * where \f$A\f$ is a positive semidefinite or negative semidefinite
		 * Hermitan matrix @see ldlt_factor
		 *
		 * @param L Triangular matrix, LDLT Cholesky factorization of A
		 * @param d The diagonal of the diagonal matrix D
		 * @param p The permuattion matrix P as a
		 * transposition sequence
		 * @param b Right-hand side array
		 * @param lower Whether to use L as the upper or lower triangular
		 *  Cholesky factorization (default:lower)
		 * @return \f$\x\f$
		 */
		template <typename T>
		SGVector<T> ldlt_solver(
		    const SGMatrix<T>& L, const SGVector<T>& d, SGVector<index_t>& p,
		    const SGVector<T>& b, const bool lower = true)
		{
			require(
			    L.num_rows == L.num_cols,
			    "Matrix dimensions ({}x{}) are not square", L.num_rows,
			    L.num_cols);
			require(
			    d.vlen == L.num_rows, "Length of vector d ({}) doesn't match "
			                          "length of diagonal of matrix L ({})",
			    d.vlen, L.num_rows);
			require(
			    p.vlen = L.num_rows, "Length of transpositions vector p ({}) "
			                         "doesn't match length of diagonal of "
			                         "matrix L ({})",
			    p.vlen, L.num_rows);

			return infer_backend(L, SGMatrix<T>(d), SGMatrix<T>(b))
			    ->ldlt_solver(L, d, p, b, lower);
		}

		/**
		 * Vector dot-product that works with generic vectors.
		 *
		 * @param a First vector
		 * @param b Second vector
		 * @return The dot product of \f$\mathbf{a}\f$ and \f$\mathbf{b}\f$,
		 * represented
		 * as \f$\sum_i a_i b_i\f$
		 */
		template <
		    typename T, typename U, typename TU = typename promote<T, U>::type,
		    typename Tag = void*>
		TU dot(const SGVector<T>& a, const SGVector<U>& b, Tag tag = {})
		{
			require(
			    a.vlen == b.vlen,
			    "Length of vector a ({}) doesn't match vector b ({}).",
			    a.vlen, b.vlen);

			static_assert(
			    std::is_same<T, U>::value ||
			        std::is_same<Tag, allow_cast>::value,
			    "LinalgNamespace::dot: Error. unmatching operands types "
			    "require allow_cast tag");

			return infer_backend(a, b)->dot(a, b);
		}

		/**
		 * Compute the eigenvalues and eigenvectors of a matrix.
		 * Note that the type of the computed values is the same
		 * as the matrix's type, i.e. for real matrices it returns
		 * only the real part of the eigenvalues/vectors.
		 *
		 * User should pass an appropriately pre-allocated memory vector
		 * to store the eigenvalues and an appropriately pre-allocated memory
		 * matrix to store the eigenvectors.
		 *
		 * @param A The matrix whose eigenvalues and eigenvectors are to be
		 * computed
		 * @param eigenvalues Eigenvalues result vector
		 * @param eigenvectors Eigenvectors result matrix
		 */
		template <typename T>
		void eigen_solver(
		    const SGMatrix<T>& A, SGVector<T>& eigenvalues,
		    SGMatrix<T>& eigenvectors)
		{
			require(
			    A.num_rows == A.num_cols, "Matrix A ({} x% d) is not square!",
			    A.num_rows, A.num_cols);
			require(
			    A.num_rows == eigenvectors.num_rows,
			    "Number of rows of A ({}) doesn't match eigenvectors' matrix "
			    "({}).",
			    A.num_rows, eigenvectors.num_rows);
			require(
			    A.num_cols == eigenvectors.num_cols,
			    "Number of columns of A ({}) doesn't match eigenvectors' "
			    "matrix ({}).",
			    A.num_cols, eigenvectors.num_cols);
			require(
			    A.num_cols == eigenvalues.vlen,
			    "Length of eigenvalues' vector doesn't match matrix A");

			infer_backend(A)->eigen_solver(A, eigenvalues, eigenvectors);
		}

		/**
		 * Compute the top-k eigenvalues and eigenvectors of a symmetric matrix.
		 *
		 * User should pass an appropriately pre-allocated memory vector
		 * to store the eigenvalues and an appropriately pre-allocated memory
		 * matrix to store the eigenvectors.
		 *
		 * @param A The matrix whose eigenvalues and eigenvectors are to be
		 * computed
		 * @param eigenvalues Eigenvalues result vector in ascending order
		 * @param eigenvectors Eigenvectors result matrix
		 * @param k number of top eigenvalues to be computed
		 * [default = 0: all eigenvalues]
		 */
		template <typename T>
		void eigen_solver_symmetric(
		    const SGMatrix<T>& A, SGVector<T>& eigenvalues,
		    SGMatrix<T>& eigenvectors, index_t k = 0)
		{

			require(
			    A.num_rows == A.num_cols, "Matrix A ({} x% d) is not square!",
			    A.num_rows, A.num_cols);

			if (k == 0)
				k = A.num_rows;
			require(
			    k > 0 && k <= A.num_rows,
			    "Invalid value of k ({}), it must be in the range 1-{}.", k,
			    A.num_rows);

			require(
			    A.num_rows == eigenvectors.num_rows,
			    "Number of rows of A ({}) doesn't match eigenvectors' matrix "
			    "({}).",
			    A.num_rows, eigenvectors.num_rows);
			require(
			    k == eigenvectors.num_cols, "Number of requested eigenvectors "
			                                "({}) doesn't match the number "
			                                "of result matrix columns ({}).",
			    k, eigenvectors.num_cols);
			require(
			    k == eigenvalues.vlen, "Length of result vector doesn't "
			                           "match the number of requested "
			                           "eigenvalues");

			infer_backend(A)->eigen_solver_symmetric(
			    A, eigenvalues, eigenvectors, k);
		}

		/** Performs the operation C = A .* B where ".*" denotes elementwise
		 * multiplication
		 * on matrix blocks.
		 *
		 * This version returns the result in-place.
		 * User should pass an appropriately pre-allocated memory matrix.
		 *
		 * This operation works with CPU backends only.
		 *
		 * @param A First matrix block
		 * @param B Second matrix block
		 * @param result Result matrix
		 * @param transpose_A whether to transpose matrix A
		 * @param transpose_B whether to transpose matrix B
		 */
		template <typename T>
		void element_prod(
		    const Block<SGMatrix<T>>& A, const Block<SGMatrix<T>>& B,
		    SGMatrix<T>& result, bool transpose_A = false,
		    bool transpose_B = false)
		{
			auto num_rows = transpose_A ? A.m_col_size : A.m_row_size;
			auto num_cols = transpose_A ? A.m_row_size : A.m_col_size;

			require(
			    (num_rows == transpose_B ? B.m_col_size : B.m_row_size) &&
			        (num_cols == transpose_B ? B.m_row_size : B.m_col_size),
			    "Dimension mismatch! A({} x {}) vs B({} x {})", A.m_row_size,
			    A.m_col_size, B.m_row_size, B.m_col_size);

			require(
			    num_rows == result.num_rows && num_cols == result.num_cols,
			    "Dimension mismatch! A({} x {}) vs result({} x {})",
			    A.m_row_size, A.m_col_size, result.num_rows, result.num_cols);

			require(
			    !result.on_gpu(),
			    "Cannot operate with matrix result on_gpu ({}) \
	 		as matrix blocks are on CPU.",
			    result.on_gpu());

			env()->linalg()->get_cpu_backend()->element_prod(
			    A, B, result, transpose_A, transpose_B);
		}

		/** Performs the operation C = A .* B where ".*" denotes elementwise
		 * multiplication
		 * on matrix blocks.
		 *
		 * This version returns the result in a newly created matrix.
		 *
		 * @param A First matrix block
		 * @param B Second matrix block
		 * @param transpose_A whether to transpose matrix A
		 * @param transpose_B whether to transpose matrix B
		 * @return The result of the operation
		 */
		template <typename T>
		SGMatrix<T> element_prod(
		    const Block<SGMatrix<T>>& A, const Block<SGMatrix<T>>& B,
		    bool transpose_A = false, bool transpose_B = false)
		{
			auto num_rows = transpose_A ? A.m_col_size : A.m_row_size;
			auto num_cols = transpose_A ? A.m_row_size : A.m_col_size;

			SGMatrix<T> result(num_rows, num_cols);

			element_prod(A, B, result, transpose_A, transpose_B);

			return result;
		}

		/** Performs the operation C = A .* B where ".*" denotes elementwise
		 * multiplication.
		 *
		 * This version returns the result in-place.
		 * User should pass an appropriately pre-allocated memory matrix
		 * Or pass one of the operands arguments (A or B) as a result
		 *
		 * @param a First matrix
		 * @param b Second matrix
		 * @param result Result matrix
		 * @param transpose_A whether to transpose matrix A
		 * @param transpose_B whether to transpose matrix B
		 */
		template <typename T>
		void element_prod(
		    const SGMatrix<T>& A, const SGMatrix<T>& B, SGMatrix<T>& result,
		    bool transpose_A = false, bool transpose_B = false)
		{
			auto num_rows = transpose_A ? A.num_cols : A.num_rows;
			auto num_cols = transpose_A ? A.num_rows : A.num_cols;

			require(
			    (num_rows == transpose_B ? B.num_cols : B.num_rows) &&
			        (num_cols == transpose_B ? B.num_rows : B.num_cols),
			    "Dimension mismatch! A({} x {}) vs B({} x {})", A.num_rows,
			    A.num_cols, B.num_rows, B.num_cols);

			require(
			    num_rows == result.num_rows && num_cols == result.num_cols,
			    "Dimension mismatch! A({} x {}) vs result({} x {})",
			    A.num_rows, A.num_cols, result.num_rows, result.num_cols);

			infer_backend(A, B, result)
			    ->element_prod(A, B, result, transpose_A, transpose_B);
		}

		/** Performs the operation C = A .* B where ".*" denotes elementwise
		 * multiplication.
		 *
		 * This version returns the result in a newly created matrix.
		 *
		 * @param A First matrix
		 * @param B Second matrix
		 * @param transpose_A whether to transpose matrix A
		 * @param transpose_B whether to transpose matrix B
		 * @return The result of the operation
		 */
		template <typename T>
		SGMatrix<T> element_prod(
		    const SGMatrix<T>& A, const SGMatrix<T>& B,
		    bool transpose_A = false, bool transpose_B = false)
		{
			auto num_rows = transpose_A ? A.num_cols : A.num_rows;
			auto num_cols = transpose_A ? A.num_rows : A.num_cols;

			SGMatrix<T> result(num_rows, num_cols);

			if (A.on_gpu())
				to_gpu(result);
			element_prod(A, B, result, transpose_A, transpose_B);

			return result;
		}

		/** Performs the operation C = A .* B where ".*" denotes elementwise
		 * multiplication.
		 *
		 * This version returns the result in a newly created vector.
		 *
		 * @param a First vector
		 * @param b Second vector
		 * @return The result of the operation
		 */
		template <typename T>
		void element_prod(
		    const SGVector<T>& a, const SGVector<T>& b, SGVector<T>& result)
		{
			require(
			    a.vlen == b.vlen, "Dimension mismatch! A({}) vs B({})",
			    a.vlen, b.vlen);
			require(
			    a.vlen == result.vlen,
			    "Dimension mismatch! A({}) vs result({})", a.vlen,
			    result.vlen);

			infer_backend(a, b)->element_prod(a, b, result);
		}

		/** Performs the operation C = A .* B where ".*" denotes elementwise
		 * multiplication.
		 *
		 * This version returns the result in a newly created vector.
		 *
		 * @param a First vector
		 * @param b Second vector
		 * @return The result of the operation
		 */
		template <typename T>
		SGVector<T> element_prod(const SGVector<T>& a, const SGVector<T>& b)
		{
			require(
			    a.vlen == b.vlen, "Dimension mismatch! A({}) vs B({})",
			    a.vlen, b.vlen);

			SGVector<T> result = a.clone();
			element_prod(a, b, result);

			return result;
		}

		/** Performs the operation B = exp(A)
		 *
		 * This version returns the result in a newly created vector or matrix.
		 *
		 * @param a Exponent vector or matrix
		 * @return The result of the operation
		 */
		template <typename T, template <typename> class Container>
		Container<T> exponent(const Container<T>& a)
		{
			Container<T> result;
			result = a.clone();

			infer_backend(a)->exponent(a, result);

			return result;
		}

		/**
		 * Method that writes the identity into a square matrix.
		 *
		 * @param a The square matrix to be set
		 */
		template <typename T>
		void identity(SGMatrix<T>& identity_matrix)
		{
			require(identity_matrix.num_rows == identity_matrix.num_cols, "Matrix is not square!");
			infer_backend(identity_matrix)->identity(identity_matrix);
		}

		/** Performs the operation of a matrix multiplies a vector \f$x = Ab\f$.
		 *
		 * This version returns the result in-place.
		 * User should pass an appropriately allocated memory matrix.
		 *
		 * @param A The matrix
		 * @param b The vector
		 * @param transpose Whether to transpose the matrix. Default false
		 * @param result Result vector
		 */
		template <typename T>
		void matrix_prod(
		    const SGMatrix<T>& A, const SGVector<T>& b, SGVector<T>& result,
		    bool transpose = false)
		{
			if (transpose)
			{
				require(
				    A.num_rows == b.vlen,
				    "Row number of Matrix A ({}) doesn't match \
			length of vector b ({}).",
				    A.num_rows, b.vlen);
				require(
				    result.vlen == A.num_cols,
				    "Length of vector result ({}) doesn't match \
			column number of Matrix A ({}).",
				    result.vlen, A.num_cols);
			}
			else
			{
				require(
				    A.num_cols == b.vlen,
				    "Column number of Matrix A ({}) doesn't match \
			length of vector b ({}).",
				    A.num_cols, b.vlen);
				require(
				    result.vlen == A.num_rows,
				    "Length of vector result ({}) doesn't match \
			row number of Matrix A ({}).",
				    result.vlen, A.num_rows);
			}

			require(
			    !(result.on_gpu() ^ A.on_gpu()), "Cannot operate with vector "
			                                     "result on_gpu ({}) and "
			                                     "vector a on_gpu ({}).",
			    result.on_gpu(), A.on_gpu());
			require(
			    !(result.on_gpu() ^ b.on_gpu()), "Cannot operate with vector "
			                                     "result on_gpu ({}) and "
			                                     "vector b on_gpu ({}).",
			    result.on_gpu(), b.on_gpu());

			infer_backend(A, SGMatrix<T>(b))
			    ->matrix_prod(A, b, result, transpose, false);
		}

		/** Performs the operation of matrix multiply a vector \f$x = Ab\f$.
		 * This version returns the result in a newly created vector.
		 *
		 * @param A The matrix
		 * @param b The vector
		 * @param transpose Whether to transpose a matrix. Default:false
		 * @return result Result vector
		 */
		template <typename T>
		SGVector<T> matrix_prod(
		    const SGMatrix<T>& A, const SGVector<T>& b, bool transpose = false)
		{
			SGVector<T> result;
			if (transpose)
			{
				require(
				    A.num_rows == b.vlen,
				    "Row number of Matrix A ({}) doesn't match \
			length of vector b ({}).",
				    A.num_rows, b.vlen);
				result = SGVector<T>(A.num_cols);
			}
			else
			{
				require(
				    A.num_cols == b.vlen,
				    "Column number of Matrix A ({}) doesn't match \
		length of vector b ({}).",
				    A.num_cols, b.vlen);
				result = SGVector<T>(A.num_rows);
			}

			if (A.on_gpu())
				to_gpu(result);

			matrix_prod(A, b, result, transpose);
			return result;
		}

		/** Performs the operation C = A * B where "*" denotes matrix
		 * multiplication.
		 *
		 * This version returns the result in-place.
		 * User should pass an appropriately allocated memory matrix
		 *
		 * @param A First matrix
		 * @param B Second matrix
		 * @param result Result matrix
		 * @param transpose_A whether to transpose matrix A
		 * @param transpose_B whether to transpose matrix B
		 */
		template <typename T>
		void matrix_prod(
		    const SGMatrix<T>& A, const SGMatrix<T>& B, SGMatrix<T>& result,
		    bool transpose_A = false, bool transpose_B = false)
		{
			require(
			    !(result.on_gpu() ^ A.on_gpu()),
			    "Cannot operate with matrix result on_gpu ({}) and \
			 matrix A on_gpu ({}).",
			    result.on_gpu(), A.on_gpu());
			require(
			    !(result.on_gpu() ^ B.on_gpu()),
			    "Cannot operate with matrix result on_gpu ({}) and \
			 matrix B on_gpu ({}).",
			    result.on_gpu(), B.on_gpu());

			if (transpose_A)
			{
				require(
				    A.num_cols == result.num_rows,
				    "Number of columns for A ({}) and \
				number of rows for result ({}) should be equal!",
				    A.num_cols, result.num_rows);
				if (transpose_B)
				{
					require(
					    A.num_rows == B.num_cols,
					    "Number of rows for A ({}) and \
					number of columns for B ({}) should be equal!",
					    A.num_rows, B.num_cols);
					require(
					    B.num_rows == result.num_cols,
					    "Number of rows for B ({}) and \
					number of columns for result ({}) should be equal!",
					    B.num_rows, result.num_cols);
				}
				else
				{
					require(
					    A.num_rows == B.num_rows,
					    "Number of rows for A ({}) and \
					number of rows for B ({}) should be equal!",
					    A.num_rows, B.num_rows);
					require(
					    B.num_cols == result.num_cols,
					    "Number of columns for B ({}) and \
					number of columns for result ({}) should be equal!",
					    B.num_cols, result.num_cols);
				}
			}
			else
			{
				require(
				    A.num_rows == result.num_rows,
				    "Number of rows for A ({}) and \
				number of rows for result ({}) should be equal!",
				    A.num_rows, result.num_rows);
				if (transpose_B)
				{
					require(
					    A.num_cols == B.num_cols,
					    "Number of columns for A ({}) and \
					number of columns for B ({}) should be equal!",
					    A.num_cols, B.num_cols);
					require(
					    B.num_rows == result.num_cols,
					    "Number of rows for B ({}) and \
					number of columns for result ({}) should be equal!",
					    B.num_rows, result.num_cols);
				}
				else
				{
					require(
					    A.num_cols == B.num_rows,
					    "Number of columns for A ({}) and \
					number of rows for B ({}) should be equal!",
					    A.num_cols, B.num_rows);
					require(
					    B.num_cols == result.num_cols,
					    "Number of columns for B ({}) and \
					number of columns for result ({}) should be equal!",
					    B.num_cols, result.num_cols);
				}
			}

			infer_backend(A, B)->matrix_prod(
			    A, B, result, transpose_A, transpose_B);
		}

		/** Performs the operation C = A * B where "*" denotes matrix
		 * multiplication.
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
		SGMatrix<T> matrix_prod(
		    const SGMatrix<T>& A, const SGMatrix<T>& B,
		    bool transpose_A = false, bool transpose_B = false)
		{
			SGMatrix<T> result;

			if (transpose_A & transpose_B)
			{
				require(
				    A.num_rows == B.num_cols, "Number of rows for A ({}) and \
				number of columns for B ({}) should be equal!",
				    A.num_rows, B.num_cols);
				result = SGMatrix<T>(A.num_cols, B.num_rows);
			}
			else if (transpose_A)
			{
				require(
				    A.num_rows == B.num_rows, "Number of rows for A ({}) and \
				number of rows for B ({}) should be equal!",
				    A.num_rows, B.num_rows);
				result = SGMatrix<T>(A.num_cols, B.num_cols);
			}
			else if (transpose_B)
			{
				require(
				    A.num_cols == B.num_cols,
				    "Number of columns for A ({}) and \
				number of columns for B ({}) should be equal!",
				    A.num_cols, B.num_cols);
				result = SGMatrix<T>(A.num_rows, B.num_rows);
			}
			else
			{
				require(
				    A.num_cols == B.num_rows,
				    "Number of columns for A ({}) and \
				number of rows for B ({}) should be equal!",
				    A.num_cols, B.num_rows);
				result = SGMatrix<T>(A.num_rows, B.num_cols);
			}

			if (A.on_gpu())
				to_gpu(result);

			matrix_prod(A, B, result, transpose_A, transpose_B);

			return result;
		}

		/**
		 * Performs the operation y = \alpha ax + \beta y
		 * This function multiplies a * x (after transposing a, if needed)
		 * and multiplies the resulting matrix by alpha. It then multiplies
		 * vector y by
		 * beta. It stores the sum of these two products in vector y
		 *
		 * @param alpha scaling factor for vector ax
		 * @param a matrix
		 * @param transpose Whether to transpose matrix a
		 * @param x vector
		 * @param beta scaling factor for vector y
		 * @param y vector
		 */
		template <typename T>
		void dgemv(
		    T alpha, const SGMatrix<T>& a, bool transpose, const SGVector<T>& x,
		    T beta, SGVector<T>& y)
		{
			auto temp_vector = matrix_prod(a, x, transpose);
			add(temp_vector, y, y, alpha, beta);
		}

		/**
		 * This function multiplies a * b and multiplies the resulting matrix by
		 * alpha.
		 * It then multiplies matrix c by beta. It stores the sum of these two
		 * products
		 * in matrix c.
		 *
		 * @param alpha scaling factor for matrix a*b
		 * @param a matrix
		 * @param b matrix
		 * @param transpose_a Whether to transpose matrix a
		 * @param transpose_b Whether to transpose matrix b
		 * @param beta scaling factor for matrix c
		 * @param c matrix
		 */
		template <typename T>
		void dgemm(
		    T alpha, const SGMatrix<T>& a, const SGMatrix<T>& b,
		    bool transpose_a, bool transpose_b, T beta, SGMatrix<T>& c)
		{
			auto temp_matrix = matrix_prod(a, b, transpose_a, transpose_b);
			add(temp_matrix, c, c, alpha, beta);
		}

		/**
		 * Returns the largest element in a vector or matrix
		 *
		 * @param a Input vector or matrix
		 * @return The largest value in the vector or matrix
		 */
		template <typename T, template <typename> class Container>
		T max(const Container<T>& a)
		{
			return infer_backend(a)->max(a);
		}

		/**
		 * Method that computes the mean of vectors or matrices composed of real
		 * numbers.
		 *
		 * @param a SGVector or SGMatrix
		 * @return The vector mean \f$\bar{a}_i\f$ or matrix mean
		 * \f$\bar{m}_{i,j}\f$
		 */
		template <typename T, template <typename> class Container>
		typename std::enable_if<!std::is_same<T, complex128_t>::value,
		                        float64_t>::type
		mean(const Container<T>& a)
		{
			require(a.size() > 0, "Vector/Matrix cannot be empty!");
			return infer_backend(a)->mean(a);
		}

		/**
		 * Method that computes the mean of vectors or matrices composed of
		 * complex numbers.
		 *
		 * @param a SGVector or SGMatrix
		 * @return The vector mean \f$\bar{a}_i\f$ or matrix mean
		 * \f$\bar{m}_{i,j}\f$
		 */
		template <template <typename> class Container>
		complex128_t mean(const Container<complex128_t>& a)
		{
			require(a.size() > 0, "Vector/Matrix cannot be empty!");
			return infer_backend(a)->mean(a);
		}

		/** Method that updates moving mean vector with new datum point.
		 *
		 * @param cma the previous moving mean
		 * @param datum new datum point
		 * @param n number of previous data points including the new datum point
		 */
		template <typename T>
		void update_mean(SGVector<T>& cma, const SGVector<T>& datum, int32_t n)
		{
			require(n > 0, "Number of data points ({}) must be at least 1", n);
			T alpha = (T)(1.0) / n;
			T beta = 1 - alpha;
			add(datum, cma, cma, alpha, beta);
		}

		/** Method that updates moving mean scalar with new datum point.
		 *
		 * @param cma the previous moving mean
		 * @param datum new datum point
		 * @param n number of previous data points including the new datum point
		 */
		template <typename T>
		void update_mean(T& cma, const T datum, int32_t n)
		{
			require(n > 0, "Number of data points ({}) must be at least 1", n);
			T alpha = (T)(1.0) / n;
			T beta = 1 - alpha;
			cma = alpha * datum + beta * cma;
		}

		/**
		 * Method that computes the standard deviation of vectors or matrices
		 * composed of real numbers.
		 *
		 * @param a SGVector or SGMatrix
		 * @return The vector mean \f$\bar{a}_i\f$ or matrix mean
		 * \f$\bar{m}_{i,j}\f$
		 */
		template <typename T>
		typename std::enable_if<
		    !std::is_same<T, complex128_t>::value, SGVector<float64_t>>::type
		std_deviation(const SGMatrix<T>& mat, bool colwise = true)
		{
			require(mat.size() > 0, "Vector/Matrix cannot be empty!");
			return infer_backend(mat)->std_deviation(mat, colwise);
		}

		/**
		 * Method that computes the euclidean norm of a vector.
		 *
		 * @param a SGVector
		 * @return The vector norm
		 */
		template <typename T>
		T norm(const SGVector<T>& a)
		{
			require(a.size() > 0, "Vector cannot be empty!");
			return std::sqrt(dot(a, a));
		}

		/**
		 * Solve the linear equations \f$Ax=b\f$ through the
		 * QR decomposition of A.
		 *
		 * @param A The matrix
		 * @param b Right-hand side vector or matrix
		 * @return \f$\x\f$
		 */
		template <typename T, template <typename> class Container>
		Container<T> qr_solver(const SGMatrix<T>& A, const Container<T>& b)
		{
			require(
			    A.num_rows == A.num_cols, "Matrix A ({} x% d) is not square!",
			    A.num_rows, A.num_cols);

			return infer_backend(A, SGMatrix<T>(b))->qr_solver(A, b);
		}

		/**
		 * Range fill a vector or matrix with start...start+len-1
		 *
		 * @param a The vector or matrix to be filled
		 * @param start Value to be assigned to the first element of vector or
		 * matrix
		 */
		template <typename T, template <typename> class Container>
		void range_fill(Container<T>& a, const T start = 0)
		{
			infer_backend(a)->range_fill(a, start);
		}

		/**
		 * Performs the operation result = alpha * a on vectors
		 * This version returns the result in-place.
		 * User should pass an appropriately pre-allocated memory matrix
		 * Or pass the operands argument a as a result
		 *
		 * @param a First vector
		 * @param alpha Scale factor
		 * @param result The vector of alpha * a
		 */
		template <typename T>
		void scale(const SGVector<T>& a, SGVector<T>& result, T alpha = 1)
		{
			require(
			    result.vlen == a.vlen,
			    "Length of vector result ({}) doesn't match vector a ({}).",
			    result.vlen, a.vlen);
			infer_backend(a, result)->scale(a, alpha, result);
		}

		/**
		 * Performs the operation result = alpha * A on matrices
		 * This version returns the result in-place.
		 * User should pass an appropriately pre-allocated memory matrix
		 * Or pass the operands argument A as a result
		 *
		 * @param A First matrix
		 * @param alpha Scale factor
		 * @param result The matrix of alpha * A
		 */
		template <typename T>
		void scale(const SGMatrix<T>& A, SGMatrix<T>& result, T alpha = 1)
		{
			require(
			    (A.num_rows == result.num_rows), "Number of rows of matrix A "
			                                     "({}) must match matrix "
			                                     "result ({}).",
			    A.num_rows, result.num_rows);
			require(
			    (A.num_cols == result.num_cols), "Number of columns of matrix "
			                                     "A ({}) must match matrix "
			                                     "result ({}).",
			    A.num_cols, result.num_cols);
			infer_backend(A, result)->scale(A, alpha, result);
		}

		/**
		 * Scales every column of the matrix A with the respective coefficient
		 * of alphas. i.e. result.column(i) = A.column(i) * alphas(i)
		 * User should pass an appropriately pre-allocated memory matrix
		 * Or pass the operands argument A as a result
		 *
		 * @param A First matrix
		 * @param alpha Scale factor
		 * @param result The scaled matrix
		 */
		template <typename T>
		void scale(
		    const SGMatrix<T>& A, SGMatrix<T>& result,
		    const SGVector<T>& alphas)
		{
			require(
			    (A.num_rows == result.num_rows),
			    "Number of rows of matrix A "
			    "({}) must match matrix "
			    "result ({}).",
			    A.num_rows, result.num_rows);
			require(
			    (A.num_cols == result.num_cols),
			    "Number of columns of matrix "
			    "A ({}) must match matrix "
			    "result ({}).",
			    A.num_cols, result.num_cols);
			require(
			    (A.num_cols == alphas.vlen),
			    "Number of columns of matrix "
			    "A ({}) must match vector "
			    "alphas ({}).",
			    A.num_cols, alphas.vlen);
			infer_backend(A, result)->scale(A, alphas, result);
		}

		/**
		 * Performs the operation B = alpha * A on vectors or matrices
		 * This version returns the result in a newly created vector or matrix.
		 *
		 * @param a First vector/matrix
		 * @param alpha Scale factor
		 * @return Vector or matrix of alpha * A
		 */
		template <typename T, template <typename> class Container>
		Container<T> scale(const Container<T>& a, T alpha = 1)
		{
			auto result = a.clone();
			scale(a, result, alpha);
			return result;
		}

		/**
		 * Set const value to vectors or matrices
		 *
		 * @param a Vector or matrix to be set
		 * @param value The value to set the vector or matrix
		 */
		template <typename T, template <typename> class Container>
		void set_const(Container<T>& a, T value)
		{
			infer_backend(a)->set_const(a, value);
		}

		/**
		 * Method that computes the sum of vectors or matrices
		 *
		 * @param a The vector or matrix whose sum has to be computed
		 * @param no_diag If true, diagonal entries are excluded from the sum.
		 * Default: false
		 * @return The vector sum \f$\sum_i a_i\f$ or matrix sum
		 * \f$\sum_{i,j}b_{i,j}\f$
		 */
		template <typename T, template <typename> class Container>
		T sum(const Container<T>& a, bool no_diag = false)
		{
			return infer_backend(a)->sum(a, no_diag);
		}

		/**
		 * Method that computes the sum of matrix blocks
		 * This operation works with CPU backends only.
		 *
		 * @param a The matrix-block whose sum of co-efficients has to be
		 * computed
		 * @param no_diag If true, diagonal entries are excluded from the sum.
		 * Default: false
		 * @return Matrix-block sum \f$\sum_{i,j}b_{i,j}\f$
		 */
		template <typename T>
		T sum(const Block<SGMatrix<T>>& a, bool no_diag = false)
		{
			return env()->linalg()->get_cpu_backend()->sum(a, no_diag);
		}

		/**
		 * Method that computes the sum of symmetric matrices
		 *
		 * @param a The symmetric matrix whose sum has to be computed
		 * @param no_diag If true, diagonal entries are excluded from the sum.
		 * Default: false
		 * @return The matrix sum \f$\sum_{i,j}b_{i,j}\f$
		 */
		template <typename T>
		T sum_symmetric(const SGMatrix<T>& a, bool no_diag = false)
		{
			require(a.num_rows == a.num_cols, "Matrix is not square!");
			return infer_backend(a)->sum_symmetric(a, no_diag);
		}

		/**
		 * Method that computes the sum of symmetric matrix blocks
		 * This operation works with CPU backends only.
		 *
		 * @param a The symmetric matrix-block whose sum has to be computed
		 * @param no_diag If true, diagonal entries are excluded from the sum.
		 * Default: false
		 * @return Symmetric matrix-block sum \f$\sum_{i,j}b_{i,j}\f$
		 */
		template <typename T>
		T sum_symmetric(const Block<SGMatrix<T>>& a, bool no_diag = false)
		{
			require(a.m_row_size == a.m_col_size, "Matrix is not square!");
			return env()->linalg()->get_cpu_backend()->sum_symmetric(a, no_diag);
		}

		/**
		 * Method that computes colwise sum of co-efficients of a dense matrix
		 *
		 * @param Mat a matrix whose colwise sum has to be computed
		 * @param no_diag If true, diagonal entries are excluded from the sum.
		 * Default:
		 * false
		 * @return The colwise sum of co-efficients computed as
		 * \f$s_j=\sum_{i}b_{i,j}\f$
		 */
		template <typename T>
		SGVector<T> colwise_sum(const SGMatrix<T>& mat, bool no_diag = false)
		{
			return infer_backend(mat)->colwise_sum(mat, no_diag);
		}

		/**
		 * Method that computes the colwise sum of matrix blocks
		 * This operation works with CPU backends only.
		 *
		 * @param a the matrix-block whose colwise sum of co-efficients has to
		 * be computed
		 * @param no_diag If true, diagonal entries are excluded from the sum.
		 * Default: false
		 * @return the colwise sum of co-efficients computed as
		 * \f$s_j=\sum_{i}b_{i,j}\f$
		 */
		template <typename T>
		SGVector<T>
		colwise_sum(const Block<SGMatrix<T>>& a, bool no_diag = false)
		{
			return env()->linalg()->get_cpu_backend()->colwise_sum(a, no_diag);
		}

		/**
		 * Method that computes rowwise sum of co-efficients of a dense matrix
		 *
		 * @param mat a matrix whose rowwise sum has to be computed
		 * @param no_diag If true, diagonal entries are excluded from the sum.
		 * Default: false
		 * @return the rowwise sum of co-efficients computed as
		 * \f$s_i=\sum_{j}m_{i,j}\f$
		 */
		template <typename T>
		SGVector<T> rowwise_sum(const SGMatrix<T>& mat, bool no_diag = false)
		{
			return infer_backend(mat)->rowwise_sum(mat, no_diag);
		}

		/**
		 * Method that computes the rowwise sum of matrix blocks
		 * This operation works with CPU backends only.
		 *
		 * @param a the matrix-block whose rowwise sum of co-efficients has to
		 * be computed
		 * @param no_diag If true, diagonal entries are excluded from the sum.
		 * Default: false
		 * @return the rowwise sum of co-efficients computed as
		 * \f$s_i=\sum_{j}m_{i,j}\f$
		 */
		template <typename T>
		SGVector<T>
		rowwise_sum(const Block<SGMatrix<T>>& a, bool no_diag = false)
		{
			return env()->linalg()->get_cpu_backend()->rowwise_sum(a, no_diag);
		}

		/**
		 * Compute the singular value decomposition \f$A = U S V^{*}\f$ of a
		 * matrix.
		 * Given the \f$m \times n\f$ matrix A with \f$r = min(m,n)\f$, user
		 * should pass
		 * an appropriately pre-allocated vector of length r and a pre-allocated
		 * matrix of size \f$m \times r\f$ for thin U or \f$m \times m\f$
		 * otherwise.
		 *
		 * @param A The matrix whose svd is to be computed
		 * @param s The vector that stores the resulting singular values
		 * @param U The matrix that stores the resulting unitary matrix U
		 * @param thin_U Whether to compute the full or thin matrix U
		 * (default:thin)
		 * @param alg Whether to compute the svd through bidiagonal divide
		 * and conquer algorithm or Jacobi's algorithm (@see SVDAlgorithm)
		 * (default: bidiagonal divide and conquer)
		 */
		template <typename T>
		void
		svd(const SGMatrix<T>& A, SGVector<T>& s, SGMatrix<T>& U,
		    bool thin_U = true,
		    SVDAlgorithm alg = SVDAlgorithm::BidiagonalDivideConquer)
		{
			auto r = Math::min(A.num_cols, A.num_rows);
			require(
			    (A.num_rows == U.num_rows),
			    "Number of rows of matrix A ({}) must match matrix U ({}).",
			    A.num_rows, U.num_rows);
			if (thin_U)
			{
				require(
				    (U.num_cols == r), "Number of columns of matrix A ({}) "
				                       "must match A's smaller dimension "
				                       "({}).",
				    A.num_cols, r);
			}
			else
			{
				require(
				    (A.num_rows == U.num_cols), "Number of rows of matrix A "
				                                "({}) must match number of "
				                                "columns of U ({}).",
				    A.num_cols, r);
			}
			require(
			    (s.vlen == r), "Length of vector s ({}) doesn't match A's "
			                   "smaller dimension ({}).",
			    s.vlen, r);

			infer_backend(A)->svd(A, s, U, thin_U, alg);
		}

		/**
		 * Method that computes the trace of square matrix.
		 *
		 * @param A The matrix whose trace has to be computed
		 * @return The trace of the matrix
		 */
		template <typename T>
		T trace(const SGMatrix<T>& A)
		{
			require(A.num_rows == A.num_cols, "Matrix is not square!");
			return infer_backend(A)->trace(A);
		}

		/**
		 * Method that computes the transpose of a matrix.
		 *
		 * @param A The matrix
		 * @return The transposed matrix
		 */
		template <typename T>
		SGMatrix<T> transpose_matrix(const SGMatrix<T>& A)
		{
			return infer_backend(A)->transpose_matrix(A);
		}

		/**
		 * Method that computes the trace of \f$AB\f$ as \f$sum(A.*B')$\f
		 *
		 * @param A The matrix A
		 * @param B The matrix B
		 * @return The trace of the product of A and B
		 */
		template <typename T>
		T trace_dot(const SGMatrix<T>& A, const SGMatrix<T>& B)
		{
			return sum(element_prod(A, B, false, true));
		}

		/**
		 * Solve the linear equations \f$Lx=b\f$,
		 * where \f$L\f$ is a triangular matrix.
		 *
		 * @param L Triangular matrix
		 * @param b Right-hand side array
		 * @param lower Whether L is upper or lower triangular (default:lower)
		 * @return \f$\x\f$
		 */
		template <typename T, template <typename> class Container>
		Container<T> triangular_solver(
		    const SGMatrix<T>& L, const Container<T>& b,
		    const bool lower = true)
		{
			return infer_backend(L, SGMatrix<T>(b))
			    ->triangular_solver(L, b, lower);
		}

		/**
		 * Method that fills with zero a vector or a matrix.
		 *
		 * @param a The vector or the matrix to be set
		 */
		template <typename T, template <typename> class Container>
		void zero(Container<T>& a)
		{
			infer_backend(a)->zero(a);
		}
	} // namespace linalg
} // namespace shogun

#endif // LINALG_NAMESPACE_H_
