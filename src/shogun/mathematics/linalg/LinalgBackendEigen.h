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

#ifndef LINALG_BACKEND_EIGEN_H__
#define LINALG_BACKEND_EIGEN_H__

#include <cmath>
#include <numeric>
#include <shogun/base/range.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/lapack.h>
#include <shogun/mathematics/linalg/LinalgBackendBase.h>
#include <shogun/mathematics/linalg/LinalgEnums.h>

namespace shogun
{
	/** @brief Linalg methods with Eigen3 backend */
	class LinalgBackendEigen : public LinalgBackendBase
	{
	public:
		/** Constructor */
		LinalgBackendEigen()
		{
			set_derived(this);
		}

		/** Eigen3 vector result = alpha*A + beta*B method */
		template <typename T>
		void
		add(const SGVector<T>& a, const SGVector<T>& b, T alpha, T beta,
		    SGVector<T>& result, derived_tag tag = derived_tag()) const;

		/** Eigen3 matrix result = alpha*A + beta*B method */
		template <typename T>
		void
		add(const SGMatrix<T>& a, const SGMatrix<T>& b, T alpha, T beta,
		    SGMatrix<T>& result, derived_tag tag = derived_tag()) const;

		/** Eigen3 add column vector method */
		template <typename T>
		void add_col_vec(
		    const SGMatrix<T>& A, index_t i, const SGVector<T>& b,
		    SGMatrix<T>& result, T alpha, T beta,
		    derived_tag tag = derived_tag()) const;

		/** Eigen3 add column vector method */
		template <typename T>
		void add_col_vec(
		    const SGMatrix<T>& A, index_t i, const SGVector<T>& b,
		    SGVector<T>& result, T alpha, T beta,
		    derived_tag tag = derived_tag()) const;

		/** Eigen3 add diagonal vector method */
		template <typename T>
		void add_diag(
		    SGMatrix<T>& A, const SGVector<T>& b, T alpha, T beta,
		    derived_tag tag = derived_tag()) const;

		/** Eigen3 add diagonal scalar method */
		template <typename T>
		void add_ridge(
		    SGMatrix<T>& A, T beta, derived_tag tag = derived_tag()) const;

		/** Eigen3 add vector to each column of matrix method */
		template <typename T>
		void add_vector(
		    const SGMatrix<T>& A, const SGVector<T>& b, SGMatrix<T>& result,
		    T alpha, T beta, derived_tag tag = derived_tag()) const;

		/** Eigen3 vector add scalar method */
		template <typename T>
		void
		add_scalar(SGVector<T>& a, T b, derived_tag tag = derived_tag()) const;

		/** Eigen3 matrix add scalar method */
		template <typename T>
		void
		add_scalar(SGMatrix<T>& a, T b, derived_tag tag = derived_tag()) const;

		/** Eigen3 center matrix method */
		template <typename T>
		void
		center_matrix(SGMatrix<T>& A, derived_tag tag = derived_tag()) const;

		/** Eigen3 Cholesky decomposition */
		template <typename T>
		SGMatrix<T> cholesky_factor(
		    const SGMatrix<T>& A, const bool lower,
		    derived_tag tag = derived_tag()) const;

		/** Eigen3 Cholesky rank one update */
		template <typename T>
		void cholesky_rank_update(
		    SGMatrix<T>& L, const SGVector<T>& b, T alpha, const bool lower,
		    derived_tag tag = derived_tag()) const;

		/** Eigen3 rank one update */
		template <typename T>
		void rank_update(
		    SGMatrix<T>& A, const SGVector<T>& b, T alpha,
		    derived_tag tag = derived_tag()) const;

		/** Eigen3 Cholesky solver */
		template <typename T>
		SGVector<T> cholesky_solver(
		    const SGMatrix<T>& L, const SGVector<T>& b, const bool lower,
		    derived_tag tag = derived_tag()) const;

		/** Eigen3 LDLT Cholesky decomposition */
		template <typename T>
		void ldlt_factor(
		    const SGMatrix<T>& A, SGMatrix<T>& L, SGVector<T>& d,
		    SGVector<index_t>& p, const bool lower,
		    derived_tag tag = derived_tag()) const;

		/** Eigen3 Pseudo inverse of self adjoint matrix */
		template <typename T>
		void pinvh(
		    const SGMatrix<T>& A, SGMatrix<T>& result,
		    derived_tag tag = derived_tag()) const;

		/** Eigen3 Pseudo inverse of self adjoint matrix */
		template <typename T>
		void pinv(
		    const SGMatrix<T>& A, SGMatrix<T>& result,
		    derived_tag tag = derived_tag()) const;

		/** Eigen3 LDLT Cholesky solver */
		template <typename T>
		SGVector<T> ldlt_solver(
		    const SGMatrix<T>& L, const SGVector<T>& d,
		    const SGVector<index_t>& p, const SGVector<T>& b, const bool lower,
		    derived_tag tag = derived_tag()) const;

		/** Eigen3 cross_entropy method
		 * The cross entropy is defined as f$ H(P,Q) = - sum_{ij}
		 * P[i,j]log(Q[i,j]) f$
		 */
		template <typename T>
		T cross_entropy(
		    const SGMatrix<T>& p, const SGMatrix<T>& q,
		    derived_tag tag = derived_tag()) const;

		/** Eigen3 vector dot-product method */
		template <typename T>
		T
		dot(const SGVector<T>& a, const SGVector<T>& b,
		    derived_tag tag = derived_tag()) const;

		/** Eigen3 eigenvalues and eigenvectors computation for real matrices.
		 */
		template <typename T>
		void eigen_solver(
		    const SGMatrix<T>& A, SGVector<T>& eigenvalues,
		    SGMatrix<T>& eigenvectors, derived_tag tag = derived_tag()) const;

		/** Eigen3 eigenvalues and eigenvectors computation for complex
		 * matrices. */
		void eigen_solver(
		    const SGMatrix<complex128_t>& A,
		    SGVector<complex128_t>& eigenvalues,
		    SGMatrix<complex128_t>& eigenvectors,
		    derived_tag tag = derived_tag()) const
		{
			typename SGMatrix<complex128_t>::EigenMatrixXtMap A_eig = A;
			typename SGMatrix<complex128_t>::EigenMatrixXtMap eigenvectors_eig =
				eigenvectors;
			typename SGVector<complex128_t>::EigenVectorXtMap eigenvalues_eig =
				eigenvalues;

			Eigen::ComplexEigenSolver<typename SGMatrix<complex128_t>::EigenMatrixXt>
				solver(A_eig);

			REQUIRE(
				solver.info() != Eigen::NumericalIssue,
				"The input contains INF or NaN values or overflow occured.\n");

			eigenvalues_eig = solver.eigenvalues();
			eigenvectors_eig = solver.eigenvectors();
		}

		/** Eigen3 eigenvalues and eigenvectors computation of symmetric
		 * matrices */
		template <typename T>
		void eigen_solver_symmetric(
		    const SGMatrix<T>& A, SGVector<T>& eigenvalues,
		    SGMatrix<T>& eigenvectors, index_t k,
		    derived_tag tag = derived_tag()) const
		{
			typename SGMatrix<T>::EigenMatrixXtMap A_eig = A;
			typename SGMatrix<T>::EigenMatrixXtMap eigenvectors_eig =
			    eigenvectors;
			typename SGVector<T>::EigenVectorXtMap eigenvalues_eig =
			    eigenvalues;

			Eigen::SelfAdjointEigenSolver<typename SGMatrix<T>::EigenMatrixXt>
			    solver(A_eig);

			/*
			 * checking for success
			 *
			 * 0: Eigen::Success. Eigenvalues computation was successful
			 * 2: Eigen::NoConvergence. Iterative procedure did not converge.
			 */
			REQUIRE(
			    solver.info() != Eigen::NoConvergence,
			    "Iterative procedure did not converge!n");

			eigenvalues_eig = solver.eigenvalues().tail(k).template cast<T>();
			eigenvectors_eig =
			    solver.eigenvectors().rightCols(k).template cast<T>();
		}

/*
 * Eigen's symmetric eigensolver uses a slower algorithm in comparison
 * to LAPACK's dsyevr, so if LAPACK is available we use it for float64 type.
 * This should be removed if eventually Eigen will provide a faster
 * symmetric eigensolver (@see
 * http://eigen.tuxfamily.org/bz/show_bug.cgi?id=522).
 */
#ifdef HAVE_LAPACK
		void eigen_solver_symmetric(
		    const SGMatrix<float64_t>& A, SGVector<float64_t>& eigenvalues,
		    SGMatrix<float64_t>& eigenvectors, index_t k,
		    derived_tag tag = derived_tag()) const
		{
			int32_t status = 0;
			int32_t n = A.num_rows;

			// dsyevr requires a vector of length n even if you want just k
			// eigenvalues
			SGVector<float64_t>::EigenVectorXt ev_eig(n);
			wrap_dsyevr(
			    'V', 'U', n, A.matrix, n, n - k + 1, n, ev_eig.data(),
			    eigenvectors.matrix, &status);

			typename SGVector<float64_t>::EigenVectorXtMap eigenvalues_eig =
			    eigenvalues;
			eigenvalues_eig = ev_eig.head(k);

			/*
			 * checking for success
			 *
			 * status == 0: successful exit
			 * status < 0: the i-th argument had an illegal value
			 * status > 0: internal error
			 */
			REQUIRE(
			    !(status < 0), "The %d-th argument han an illegal value.",
			    -status)
			REQUIRE(!(status > 0), "Internal error.")
		}
#endif

		/** Eigen3 matrix in-place elementwise product method */
		template <typename T>
		void element_prod(
		    const SGMatrix<T>& a, const SGMatrix<T>& b, SGMatrix<T>& result,
		    bool transpose_A, bool transpose_B,
		    derived_tag tag = derived_tag()) const;

		/** Eigen3 matrix block in-place elementwise product method */
		template <typename T>
		void element_prod(
		    const linalg::Block<SGMatrix<T>>& a,
		    const linalg::Block<SGMatrix<T>>& b, SGMatrix<T>& result,
		    bool transpose_A, bool transpose_B,
		    derived_tag tag = derived_tag()) const;

		/** Eigen3 vector in-place elementwise product method */
		template <typename T>
		void element_prod(
		    const SGVector<T>& a, const SGVector<T>& b, SGVector<T>& result,
		    derived_tag tag = derived_tag()) const;

		/** Eigen3 vector exponent method */
		template <typename T>
		void exponent(
		    const SGVector<T>& a, SGVector<T>& result,
		    derived_tag tag = derived_tag()) const;

		/** Eigen3 matrix exponent method */
		template <typename T>
		void exponent(
		    const SGMatrix<T>& a, SGMatrix<T>& result,
		    derived_tag tag = derived_tag()) const;

		/** Eigen3 set matrix to identity method */
		template <typename T>
		void identity(
		    SGMatrix<T>& identity_matrix,
		    derived_tag tag = derived_tag()) const;

		/** Eigen3 logistic method. Calculates f(x) = 1/(1+exp(-x)) */
		template <typename T>
		void logistic(
		    const SGMatrix<T>& a, SGMatrix<T>& result,
		    derived_tag tag = derived_tag()) const;

		/** Eigen3 matrix * vector in-place product method */
		template <typename T>
		void matrix_prod(
		    const SGMatrix<T>& a, const SGVector<T>& b, SGVector<T>& result,
		    bool transpose, bool transpose_B = false,
		    derived_tag tag = derived_tag()) const;

		/** Eigen3 matrix in-place product method */
		template <typename T>
		void matrix_prod(
		    const SGMatrix<T>& a, const SGMatrix<T>& b, SGMatrix<T>& result,
		    bool transpose_A, bool transpose_B,
		    derived_tag tag = derived_tag()) const;

		/** Return the largest element in the vector with Eigen3 library */
		template <typename T>
		T max(const SGVector<T>& vec, derived_tag tag = derived_tag()) const;

		/** Return the largest element in the matrix with Eigen3 library */
		template <typename T>
		T max(const SGMatrix<T>& mat, derived_tag tag = derived_tag()) const;

		/** Real eigen3 vector and matrix mean method */
		template <typename T, template <typename> class Container>
		typename std::enable_if<
		    !std::is_same<T, complex128_t>::value, float64_t>::type
		mean(const Container<T>& a, derived_tag tag = derived_tag()) const;

		/** Complex eigen3 vector and matrix mean method */
		template <template <typename> class Container>
		complex128_t mean(
		    const Container<complex128_t>& a,
		    derived_tag tag = derived_tag()) const;

		/** Real eigen3 vector and matrix standard deviation method */
		template <typename T>
		typename std::enable_if<
		    !std::is_same<T, complex128_t>::value, SGVector<float64_t>>::type
		std_deviation(
		    const SGMatrix<T>& mat, bool colwise = true,
		    derived_tag tag = derived_tag()) const;

		/** Eigen3 multiply_by_logistic_derivative method
		 * Performs the operation C(i,j) = C(i,j) * A(i,j) * (1.0-A(i,j)) for
		 * all i
		 * and j
		 */
		template <typename T>
		void multiply_by_logistic_derivative(
		    const SGMatrix<T>& a, SGMatrix<T>& result,
		    derived_tag tag = derived_tag()) const;

		/** Eigen3 multiply_by_rectified_linear_derivative method
		 * Performs the operation C(i,j) = C(i,j) * (A(i,j)!=0) for all i and j
		 */
		template <typename T>
		void multiply_by_rectified_linear_derivative(
		    const SGMatrix<T>& a, SGMatrix<T>& result,
		    derived_tag tag = derived_tag()) const;

		/** Eigen3 vector QR solver. */
		template <typename T>
		SGVector<T> qr_solver(
		    const SGMatrix<T>& A, const SGVector<T>& b,
		    derived_tag tag = derived_tag()) const;

		/** Eigen3 matrix QR solver. */
		template <typename T>
		SGMatrix<T> qr_solver(
		    const SGMatrix<T>& A, const SGMatrix<T> b,
		    derived_tag tag = derived_tag()) const;

		/** Range fill a vector or matrix with start...start+len-1. */
		template <typename T, template <typename> class Container>
		void range_fill(
		    Container<T>& a, const T start,
		    derived_tag tag = derived_tag()) const;

		/** Applies the elementwise rectified linear function f(x) = max(0,x) */
		template <typename T>
		void rectified_linear(
		    const SGMatrix<T>& a, SGMatrix<T>& result,
		    derived_tag tag = derived_tag()) const;

		/** Eigen3 vector inplace scale method: result = alpha * A */
		template <typename T>
		void scale(
		    const SGVector<T>& a, T alpha, SGVector<T>& result,
		    derived_tag tag = derived_tag()) const;

		/** Eigen3 matrix inplace scale method: result = alpha * A */
		template <typename T>
		void scale(
		    const SGMatrix<T>& a, T alpha, SGMatrix<T>& result,
		    derived_tag tag = derived_tag()) const;

		/** Eigen3 set const method */
		template <typename T>
		void set_const(
		    SGVector<T>& a, T value, derived_tag tag = derived_tag()) const;

		/** Eigen3 set matrix to const */
		template <typename T>
		void set_const(
		    SGMatrix<T>& a, T value, derived_tag tag = derived_tag()) const;

		/** Eigen3 softmax method */
		template <typename T, template <typename> class Container>
		void softmax(Container<T>& a, derived_tag tag = derived_tag()) const;

		/** Eigen3 squared error method
		 * The squared error is defined as f$ E(P,Q) = frac{1}{2} sum_{ij}
		 * (P[i,j]-Q[i,j])^2 f$
		 */
		template <typename T>
		T squared_error(
		    const SGMatrix<T>& p, const SGMatrix<T>& q,
		    derived_tag tag = derived_tag()) const;

		/** Eigen3 vector sum method */
		template <typename T>
		T
		sum(const SGVector<T>& vec, bool no_diag = false,
		    derived_tag tag = derived_tag()) const;

		/** Eigen3 matrix sum method */
		template <typename T>
		T
		sum(const SGMatrix<T>& mat, bool no_diag = false,
		    derived_tag tag = derived_tag()) const;

		/** Eigen3 matrix block sum method */
		template <typename T>
		T
		sum(const linalg::Block<SGMatrix<T>>& mat, bool no_diag = false,
		    derived_tag tag = derived_tag()) const;

		/** Eigen3 symmetric matrix sum method */
		template <typename T>
		T sum_symmetric(
		    const SGMatrix<T>& mat, bool no_diag = false,
		    derived_tag tag = derived_tag()) const;

		/** Eigen3 symmetric matrix block sum method */
		template <typename T>
		T sum_symmetric(
		    const linalg::Block<SGMatrix<T>>& mat, bool no_diag = false,
		    derived_tag tag = derived_tag()) const;

		/** Eigen3 matrix colwise sum method */
		template <typename T>
		SGVector<T> colwise_sum(
		    const SGMatrix<T>& mat, bool no_diag,
		    derived_tag tag = derived_tag()) const;

		/** Eigen3 matrix block colwise sum method */
		template <typename T>
		SGVector<T> colwise_sum(
		    const linalg::Block<SGMatrix<T>>& mat, bool no_diag,
		    derived_tag tag = derived_tag()) const;

		/** Eigen3 matrix rowwise sum method */
		template <typename T>
		SGVector<T> rowwise_sum(
		    const SGMatrix<T>& mat, bool no_diag,
		    derived_tag tag = derived_tag()) const;

		/** Eigen3 matrix block rowwise sum method */
		template <typename T>
		SGVector<T> rowwise_sum(
		    const linalg::Block<SGMatrix<T>>& mat, bool no_diag,
		    derived_tag tag = derived_tag()) const;

		/** Eigen3 compute svd method */
		template <typename T>
		void
		svd(const SGMatrix<T>& A, SGVector<T>& s, SGMatrix<T>& U, bool thin_U,
		    linalg::SVDAlgorithm alg, derived_tag tag = derived_tag()) const;

		/** Eigen3 compute trace method */
		template <typename T>
		T trace(const SGMatrix<T>& A, derived_tag tag = derived_tag()) const;

		/** Eigen3 transpose matrix method */
		template <typename T>
		SGMatrix<T> transpose_matrix(
		    const SGMatrix<T>& A, derived_tag tag = derived_tag()) const;

		/** Eigen3 triangular solver method */
		template <typename T>
		SGMatrix<T> triangular_solver(
		    const SGMatrix<T>& L, const SGMatrix<T>& b, const bool lower,
		    derived_tag tag = derived_tag()) const;

		/** Eigen3 triangular solver method */
		template <typename T>
		SGVector<T> triangular_solver(
		    const SGMatrix<T>& L, const SGVector<T>& b, const bool lower,
		    derived_tag tag = derived_tag()) const;

		/** Eigen3 set vector to zero method */
		template <typename T>
		void zero(SGVector<T>& a, derived_tag tag = derived_tag()) const;

		/** Eigen3 set matrix to zero method */
		template <typename T>
		void zero(SGMatrix<T>& a, derived_tag tag = derived_tag()) const;
	};

#include <shogun/mathematics/linalg/backend/eigen/BasicOps.h>
#include <shogun/mathematics/linalg/backend/eigen/Decompositions.h>
#include <shogun/mathematics/linalg/backend/eigen/EigenSolvers.h>
#include <shogun/mathematics/linalg/backend/eigen/Functions.h>
#include <shogun/mathematics/linalg/backend/eigen/Misc.h>
#include <shogun/mathematics/linalg/backend/eigen/Solvers.h>
#include <shogun/mathematics/linalg/backend/eigen/SpecialPurposes.h>
} // namespace shogun

#endif // LINALG_BACKEND_EIGEN_H__
