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

#ifndef LINALG_BACKEND_BASE_H__
#define LINALG_BACKEND_BASE_H__

#include <memory>
#include <shogun/base/variant.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/common.h>
#include <shogun/lib/config.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/GPUMemoryBase.h>
#include <shogun/mathematics/linalg/LinalgEnums.h>
#include <shogun/mathematics/linalg/internal/Block.h>
#include <type_traits>

namespace shogun
{
	class LinalgBackendEigen;
	class LinalgBackendViennaCL;

#define BODY(METHOD, ...)                                                      \
	if (std::is_same<Tag, derived_tag>::value)                                 \
	{                                                                          \
		SG_SNOTIMPLEMENTED;                                                    \
	}                                                                          \
	return visit(                                                              \
	    [&](auto&& backend) {                                                  \
		    return backend->METHOD(__VA_ARGS__, derived_tag());                \
	    },                                                                     \
	    m_derived)

	/** @brief Base interface of generic linalg methods
	 * and generic memory transfer methods.
	 */
	class LinalgBackendBase
	{
	public:
		~LinalgBackendBase() = default;

		struct base_tag
		{
		};
		struct derived_tag
		{
		};

		template <typename Derived>
		void set_derived(Derived* d)
		{
			m_derived = d;
		}

		/**
		 * Wrapper method of add operation the operation
		 * result = alpha*a + beta*b.
		 *
		 * @see linalg::add
		 */
		template <
		    template <typename> class Container, typename Type,
		    typename Tag = base_tag>
		void
		add(const Container<Type>& a, const Container<Type>& b, Type alpha,
		    Type beta, Container<Type>& result, Tag tag = Tag()) const
		{
			BODY(add, a, b, alpha, beta, result);
		}

		/**
		 * Wrapper method of add column vector result = alpha*A.col(i) + beta*b.
		 *
		 * @see linalg::add_col_vec
		 */
		template <
		    template <typename> class Container, typename Type,
		    typename Tag = base_tag>
		void add_col_vec(
		    const SGMatrix<Type>& A, index_t i, const SGVector<Type>& b,
		    Container<Type>& result, Type alpha, Type beta,
		    Tag tag = Tag()) const
		{
			BODY(add_col_vec, A, i, b, result, alpha, beta);
		}

		/**
		 * Wrapper method of add diagonal vector A.diagonal = alpha * A.diagonal
		 * + beta * b.
		 *
		 * @see linalg::add_diag
		 */
		template <typename Type, typename Tag = base_tag>
		void add_diag(
		    SGMatrix<Type>& A, const SGVector<Type>& b, Type alpha, Type beta,
		    Tag tag = Tag()) const
		{
			BODY(add_diag, A, b, alpha, beta);
		}

		/**
		 * Wrapper method of add diagonal vector A.diagonal = A.diagonal + beta
		 * * b.
		 *
		 * @see linalg::add_ridge
		 */
		template <typename Type, typename Tag = base_tag>
		void add_ridge(SGMatrix<Type>& A, Type beta, Tag tag = Tag()) const
		{
			BODY(add_ridge, A, beta);
		}

		/**
		 * Wrapper method of add vector to each column of matrix.
		 *
		 * @see linalg::add_vector
		 */
		template <typename Type, typename Tag = base_tag>
		void add_vector(
		    const SGMatrix<Type>& A, const SGVector<Type>& b,
		    SGMatrix<Type>& result, Type alpha, Type beta,
		    Tag tag = Tag()) const
		{
			BODY(add_vector, A, b, result, alpha, beta);
		}

		/**
		 * Wrapper method of add scalar operation.
		 *
		 * @see linalg::add_scalar
		 */
		template <
		    template <typename> class Container, typename Type,
		    typename Tag = base_tag>
		void add_scalar(Container<Type>& a, Type b, Tag tag = Tag()) const
		{
			BODY(add_scalar, a, b);
		}

		/**
		 * Wrapper method of center matrix operation.
		 *
		 * @see linalg::center_matrix
		 */
		template <
		    template <typename> class Container, typename Type,
		    typename Tag = base_tag>
		void center_matrix(Container<Type>& A, Tag tag = Tag()) const
		{
			BODY(center_matrix, A);
		}

		/**
		 * Wrapper method of Cholesky decomposition.
		 *
		 * @see linalg::cholesky_factor
		 */
		template <
		    template <typename> class Container, typename Type,
		    typename Tag = base_tag>
		Container<Type> cholesky_factor(
		    const Container<Type>& A, const bool lower, Tag tag = Tag()) const
		{
			BODY(cholesky_factor, A, lower);
		}

		/**
		 * Wrapper triangular solver with Choleksy decomposition.
		 *
		 * @see linalg::cholesky_solver
		 */
		template <
		    template <typename> class Container, typename Type,
		    typename Tag = base_tag>
		SGVector<Type> cholesky_solver(
		    const Container<Type>& L, const SGVector<Type>& b, const bool lower,
		    Tag tag = Tag()) const
		{
			BODY(cholesky_solver, L, b, lower);
		}

		/**
		 * Wrapper for rank one update of Cholesky decomposition
		 *
		 * @see linalg::cholesky_factor
		 */
		template <
		    template <typename> class Container, typename Type,
		    typename Tag = base_tag>
		void cholesky_rank_update(
		    Container<Type>& L, const SGVector<Type>& b, Type alpha,
		    const bool lower, Tag tag = Tag()) const
		{
			BODY(cholesky_rank_update, L, b, alpha, lower);
		}

		/**
		 * Wrapper for rank one update of a square matrix
		 */
		template <
		    template <typename> class Container, typename Type,
		    typename Tag = base_tag>
		void rank_update(
		    Container<Type>& A, const SGVector<Type>& b, Type alpha,
		    Tag tag = Tag()) const
		{
			BODY(rank_update, A, b, alpha);
		}

		/**
		 * Wrapper method of LDLT Cholesky decomposition
		 *
		 * @see linalg::ldlt_factor
		 */
		template <
		    template <typename> class Container, typename Type,
		    typename Tag = base_tag>
		void ldlt_factor(
		    const Container<Type>& A, Container<Type>& L, SGVector<Type>& d,
		    SGVector<index_t>& p, const bool lower, Tag tag = Tag()) const
		{
			BODY(ldlt_factor, A, L, d, p, lower);
		}

		/**
		 * Wrapper method of pseudo inverse for self adjoint matrices
		 *
		 * @see linalg::pinvh
		 */
		template <typename Type, typename Tag = base_tag>
		void pinvh(
		    const SGMatrix<Type>& A, SGMatrix<Type>& result,
		    Tag tag = Tag()) const
		{
			BODY(pinvh, A, result);
		}

		/**
		 * Wrapper method of pseudo inverse
		 *
		 * @see linalg::pinvh
		 */
		template <typename Type, typename Tag = base_tag>
		void pinv(
		    const SGMatrix<Type>& A, SGMatrix<Type>& result,
		    Tag tag = Tag()) const
		{
			BODY(pinv, A, result);
		}

		/**
		 * Wrapper method of LDLT Cholesky solver
		 *
		 * @see linalg::ldlt_solver
		 */
		template <
		    template <typename> class Container, typename Type,
		    typename Tag = base_tag>
		SGVector<Type> ldlt_solver(
		    const Container<Type>& L, const SGVector<Type>& d,
		    const SGVector<index_t>& p, const SGVector<Type>& b,
		    const bool lower, Tag tag = Tag()) const
		{
			BODY(ldlt_solver, L, d, p, b, lower);
		}

		/**
		 * Wrapper method of cross entropy.
		 *
		 * @see linalg::cross_entropy
		 */
		template <
		    template <typename> class Container, typename Type,
		    typename Tag = base_tag>
		Type cross_entropy(
		    const Container<Type>& P, const Container<Type>& Q,
		    Tag tag = Tag()) const
		{
			BODY(cross_entropy, P, Q);
		}

		/**
		 * Wrapper method of vector dot-product that works with generic vectors.
		 *
		 * @see linalg::dot
		 */
		template <
		    template <typename> class Container, typename Type,
		    typename Tag = base_tag>
		Type
		dot(const Container<Type>& a, const Container<Type>& b,
		    Tag tag = Tag()) const
		{
			BODY(dot, a, b);
		}

		/**
		 * Wrapper method of eigenvalues and eigenvectors computation.
		 *
		 * @see linalg::eigen_solver
		 */
		template <
		    template <typename> class Container, typename Type,
		    typename Tag = base_tag>
		void eigen_solver(
		    const Container<Type>& A, SGVector<Type>& eigenvalues,
		    SGMatrix<Type>& eigenvectors, Tag tag = Tag()) const
		{
			BODY(eigen_solver, A, eigenvalues, eigenvectors);
		}

		/**
		 * Wrapper method of eigenvalues and eigenvectors computation
		 * for symmetric matrices.
		 *
		 * @see linalg::eigen_solver_symmetric
		 */
		template <
		    template <typename> class Container, typename Type,
		    typename Tag = base_tag>
		void eigen_solver_symmetric(
		    const Container<Type>& A, SGVector<Type>& eigenvalues,
		    SGMatrix<Type>& eigenvectors, index_t k, Tag tag = Tag()) const
		{
			BODY(eigen_solver_symmetric, A, eigenvalues, eigenvectors, k);
		}

		/**
		 * Wrapper method of in-place vector elementwise product.
		 *
		 * @see linalg::element_prod
		 */
		template <
		    template <typename> class Container, typename Type,
		    typename Tag = base_tag>
		void element_prod(
		    const Container<Type>& a, const Container<Type>& b,
		    Container<Type>& result, Tag tag = Tag()) const
		{
			BODY(element_prod, a, b, result);
		}

		/**
		 * Wrapper method of in-place matrix elementwise product.
		 *
		 * @see linalg::element_prod
		 */
		template <
		    template <typename> class Container, typename Type,
		    typename Tag = base_tag>
		void element_prod(
		    const Container<Type>& a, const Container<Type>& b,
		    Container<Type>& result, bool transpose_A, bool transpose_B,
		    Tag tag = Tag()) const
		{
			BODY(element_prod, a, b, result, transpose_A, transpose_B);
		}

		/**
		 * Wrapper method of in-place matrix block elementwise product.
		 *
		 * @see linalg::element_prod
		 */
		template <
		    template <typename> class Container, typename Type,
		    typename Tag = base_tag>
		void element_prod(
		    const linalg::Block<Container<Type>>& a,
		    const linalg::Block<Container<Type>>& b, Container<Type>& result,
		    bool transpose_A, bool transpose_B, Tag tag = Tag()) const
		{
			BODY(element_prod, a, b, result, transpose_A, transpose_B);
		}

		/**
		 * Wrapper method of in-place exponent method.
		 *
		 * @see linalg::exponent
		 */
		template <
		    template <typename> class Container, typename Type,
		    typename Tag = base_tag>
		void exponent(
		    const Container<Type>& a, Container<Type>& result,
		    Tag tag = Tag()) const
		{
			BODY(exponent, a, result);
		}

		/**
		 * Wrapper method of set matrix to identity.
		 *
		 * @see linalg::identity
		 */
		template <
		    template <typename> class Container, typename Type,
		    typename Tag = base_tag>
		void identity(Container<Type>& identity_matrix, Tag tag = Tag()) const
		{
			BODY(identity, identity_matrix);
		}

		/**
		 * Wrapper method of logistic function f(x) = 1/(1+exp(-x))
		 *
		 * @see linalg::logistic
		 */
		template <
		    template <typename> class Container, typename Type,
		    typename Tag = base_tag>
		void logistic(
		    const Container<Type>& a, Container<Type>& result,
		    Tag tag = Tag()) const
		{
			BODY(logistic, a, result);
		}

		/**
		 * Wrapper method of matrix product method.
		 *
		 * @see linalg::matrix_prod
		 */
		template <
		    template <typename> class Container, typename Type,
		    typename Tag = base_tag>
		void matrix_prod(
		    const SGMatrix<Type>& a, const Container<Type>& b,
		    Container<Type>& result, bool transpose_A, bool transpose_B,
		    Tag tag = Tag()) const
		{
			BODY(matrix_prod, a, b, result, transpose_A, transpose_B);
		}

		/**
		 * Wrapper method of max method. Return the largest element in a
		 * vector or matrix.
		 *
		 * @see linalg::max
		 */
		template <
		    template <typename> class Container, typename Type,
		    typename Tag = base_tag>
		Type max(const Container<Type>& a, Tag tag = Tag()) const
		{
			BODY(max, a);
		}

		/**
		 * Wrapper method that computes mean of SGVectors and SGMatrices
		 * that are composed of real numbers.
		 *
		 * @see linalg::mean
		 */
		template <
		    template <typename> class Container, typename Type,
		    typename Tag = base_tag>
		float64_t mean(const Container<Type>& a, Tag tag = Tag()) const
		{
			BODY(mean, a);
		}

		/**
		 * Wrapper method that computes mean of SGVectors and SGMatrices
		 * that are composed of complex numbers.
		 *
		 * @see linalg::mean
		 */
		template <
		    template <typename> class Container, typename Type,
		    typename Tag = base_tag>
		complex128_t
		mean(const Container<complex128_t>& a, Tag tag = Tag()) const
		{
			BODY(mean, a);
		}

		/**
		 * Wrapper method that computes mean of SGVectors and SGMatrices
		 * that are composed of real numbers.
		 *
		 * @see linalg::std_deviation
		 */
		template <
		    template <typename> class Container, typename Type,
		    typename Tag = base_tag>
		SGVector<float64_t> std_deviation(
		    const Container<Type>& a, bool colwise, Tag tag = Tag()) const
		{
			BODY(std_deviation, a, colwise);
		}

		/**
		 * Wrapper method of multiply_by_logistic_derivative function f(x) =
		 * 1/(1+exp(-x))
		 *
		 * @see linalg::multiply_by_logistic_derivative
		 */
		template <
		    template <typename> class Container, typename Type,
		    typename Tag = base_tag>
		void multiply_by_logistic_derivative(
		    const Container<Type>& a, Container<Type>& result,
		    Tag tag = Tag()) const
		{
			BODY(multiply_by_logistic_derivative, a, result);
		}

		/**
		 * Wrapper method of multiply_by_rectified_linear_derivative
		 *
		 * @see linalg::multiply_by_rectified_linear_derivative
		 */
		template <
		    template <typename> class Container, typename Type,
		    typename Tag = base_tag>
		void multiply_by_rectified_linear_derivative(
		    const Container<Type>& a, Container<Type>& result,
		    Tag tag = Tag()) const
		{
			BODY(multiply_by_rectified_linear_derivative, a, result);
		}

		/**
		 * Wrapper method that range fills a vector of matrix.
		 *
		 * @see linalg::range_fill
		 */
		template <
		    template <typename> class Container, typename Type,
		    typename Tag = base_tag>
		void
		range_fill(Container<Type>& a, const Type start, Tag tag = Tag()) const
		{
			BODY(range_fill, a, start);
		}

		/**
		 * Wrapper method of rectified_linear method f(x) = max(0, x)
		 *
		 * @see linalg::rectified_linear
		 */
		template <
		    template <typename> class Container, typename Type,
		    typename Tag = base_tag>
		void rectified_linear(
		    const Container<Type>& a, Container<Type>& result,
		    Tag tag = Tag()) const
		{
			BODY(rectified_linear, a, result);
		}

		/**
		 * Wrapper method that solves a system of linear equations
		 * using QR decomposition.
		 *
		 * @see linalg::qr_solver
		 */
		template <
		    template <typename> class Container, typename Type,
		    typename Tag = base_tag>
		Container<Type> qr_solver(
		    const SGMatrix<Type>& A, const Container<Type>& b,
		    Tag tag = Tag()) const
		{
			BODY(qr_solver, A, b);
		}

		/**
		 * Wrapper method of scale operation the operation result = alpha*A.
		 *
		 * @see linalg::scale
		 */
		template <
		    template <typename> class Container, typename Type,
		    typename Tag = base_tag>
		void scale(
		    const Container<Type>& a, Type alpha, Container<Type>& result,
		    Tag tag = Tag()) const
		{
			BODY(scale, a, alpha, result);
		}

		/**
		 * Wrapper method that sets const values to vectors or matrices.
		 *
		 * @see linalg::set_const
		 */
		template <
		    template <typename> class Container, typename Type,
		    typename Tag = base_tag>
		void
		set_const(Container<Type>& a, const Type value, Tag tag = Tag()) const
		{
			BODY(set_const, a, value);
		}

		/**
		 * Wrapper method of sum that works with generic vectors or
		 * matrices.
		 *
		 * @see linalg::sum
		 */
		template <
		    template <typename> class Container, typename Type,
		    typename Tag = base_tag>
		Type sum(const Container<Type>& a, bool no_diag, Tag tag = Tag()) const
		{
			BODY(sum, a, no_diag);
		}

		/**
		 * Wrapper method of softmax method.
		 *
		 * @see linalg::softmax
		 */
		template <
		    template <typename> class Container, typename Type,
		    typename Tag = base_tag>
		void softmax(Container<Type>& a, Tag tag = Tag()) const
		{
			BODY(softmax, a);
		}

		/**
		 * Wrapper method of squared error method.
		 *
		 * @see linalg::squared_error
		 */
		template <
		    template <typename> class Container, typename Type,
		    typename Tag = base_tag>
		Type squared_error(
		    const Container<Type>& P, const Container<Type>& Q,
		    Tag tag = Tag()) const
		{
			BODY(squared_error, P, Q);
		}

		/**
		 * Wrapper method of sum that works with matrix blocks.
		 *
		 * @see linalg::sum
		 */
		template <
		    template <typename> class Container, typename Type,
		    typename Tag = base_tag>
		Type
		sum(const linalg::Block<Container<Type>>& a, bool no_diag,
		    Tag tag = Tag()) const
		{
			BODY(sum, a, no_diag);
		}

		/**
		 * Wrapper method of sum that works with symmetric matrices.
		 *
		 * @see linalg::sum_symmetric
		 */
		template <
		    template <typename> class Container, typename Type,
		    typename Tag = base_tag>
		Type sum_symmetric(
		    const Container<Type>& a, bool no_diag, Tag tag = Tag()) const
		{
			BODY(sum_symmetric, a, no_diag);
		}

		/**
		 * Wrapper method of sum that works with symmetric matrix blocks.
		 *
		 * @see linalg::sum
		 */
		template <
		    template <typename> class Container, typename Type,
		    typename Tag = base_tag>
		Type sum_symmetric(
		    const linalg::Block<Container<Type>>& a, bool no_diag,
		    Tag tag = Tag()) const
		{
			BODY(sum_symmetric, a, no_diag);
		}

		/**
		 * Wrapper method of matrix rowwise sum that works with dense
		 * matrices.
		 *
		 * @see linalg::colwise_sum
		 */
		template <
		    template <typename> class Container, typename Type,
		    typename Tag = base_tag>
		SGVector<Type> colwise_sum(
		    const Container<Type>& a, bool no_diag, Tag tag = Tag()) const
		{
			BODY(colwise_sum, a, no_diag);
		}

		/**
		 * Wrapper method of matrix colwise sum that works with dense
		 * matrices.
		 *
		 * @see linalg::colwise_sum
		 */
		template <
		    template <typename> class Container, typename Type,
		    typename Tag = base_tag>
		SGVector<Type> colwise_sum(
		    const linalg::Block<Container<Type>>& a, bool no_diag,
		    Tag tag = Tag()) const
		{
			BODY(colwise_sum, a, no_diag);
		}

		/**
		 * Wrapper method of matrix rowwise sum that works with dense
		 * matrices.
		 *
		 * @see linalg::rowwise_sum
		 */
		template <
		    template <typename> class Container, typename Type,
		    typename Tag = base_tag>
		SGVector<Type> rowwise_sum(
		    const Container<Type>& a, bool no_diag, Tag tag = Tag()) const
		{
			BODY(rowwise_sum, a, no_diag);
		}

		/**
		 * Wrapper method of matrix rowwise sum that works with dense
		 * matrices.
		 *
		 * @see linalg::rowwise_sum
		 */
		template <
		    template <typename> class Container, typename Type,
		    typename Tag = base_tag>
		SGVector<Type> rowwise_sum(
		    const linalg::Block<Container<Type>>& a, bool no_diag,
		    Tag tag = Tag()) const
		{
			BODY(rowwise_sum, a, no_diag);
		}

		/**
		 * Wrapper method of svd computation.
		 *
		 * @see linalg::svd
		 */
		template <
		    template <typename> class Container, typename Type,
		    typename Tag = base_tag>
		void
		svd(const Container<Type>& A, SGVector<Type> s, SGMatrix<Type> U,
		    bool thin_U, linalg::SVDAlgorithm alg, Tag tag = Tag()) const
		{
			BODY(svd, A, s, U, thin_U, alg);
		}

		/**
		 * Wrapper method of trace computation.
		 *
		 * @see linalg::trace
		 */
		template <
		    template <typename> class Container, typename Type,
		    typename Tag = base_tag>
		Type trace(const Container<Type>& A, Tag tag = Tag()) const
		{
			BODY(trace, A);
		}

		/**
		 * Wrapper method of trace computation.
		 *
		 * @see linalg::transpose_matrix
		 */
		template <
		    template <typename> class Container, typename Type,
		    typename Tag = base_tag>
		Container<Type>
		transpose_matrix(const Container<Type>& A, Tag tag = Tag()) const
		{
			BODY(transpose_matrix, A);
		}

		/**
		 * Wrapper method of triangular solver.
		 *
		 * @see linalg::triangular_solver
		 */
		template <
		    template <typename> class Container, typename Type,
		    typename Tag = base_tag>
		Container<Type> triangular_solver(
		    const SGMatrix<Type>& L, const Container<Type>& b,
		    const bool lower = true, Tag tag = Tag()) const
		{
			BODY(triangular_solver, L, b, lower);
		}

		/**
		 * Wrapper method of set vector or matrix to zero.
		 *
		 * @see linalg::zero
		 */
		template <
		    template <typename> class Container, typename Type,
		    typename Tag = base_tag>
		void zero(Container<Type>& a, Tag tag = Tag()) const
		{
			BODY(zero, a);
		}

		/**
		 * Wrapper method of Transferring data to GPU memory.
		 * Does nothing if no GPU backend registered.
		 *
		 * @see linalg::to_gpu
		 */
		template <
		    template <typename> class Container, typename Type,
		    typename Tag = base_tag>
		GPUMemoryBase<Type>*
		to_gpu(const Container<Type>& a, Tag tag = Tag()) const
		{
			BODY(to_gpu, a);
		}

		/**
		 * Wrapper method of fetching data from GPU memory.
		 *
		 * @see linalg::from_gpu
		 */
		template <
		    template <typename> class Container, typename Type,
		    typename Tag = base_tag>
		void
		from_gpu(const Container<Type>& a, Type* data, Tag tag = Tag()) const
		{
			BODY(from_gpu, a, data);
		}

	private:
		variant<LinalgBackendEigen*, LinalgBackendViennaCL*> m_derived;
	};
	#undef BODY
} // namespace shogun

#endif // LINALG_BACKEND_BASE_H__
