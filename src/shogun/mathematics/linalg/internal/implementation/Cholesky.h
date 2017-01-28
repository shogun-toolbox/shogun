/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2016 Kunal Arora
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 */

#ifndef CHOLESKY_IMPL_H_
#define CHOLESKY_IMPL_H_

#include <shogun/lib/config.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/eigen3.h>
#include <numeric>

namespace shogun
{

namespace linalg
{

namespace implementation
{

/**
 * @brief Generic class which is specialized for different backends to compute the cholesky decomposition of a dense matrix
 */
template <Backend backend, class Matrix>
struct cholesky
{

	/**Scalar type */
	typedef typename Matrix::Scalar T;

	/** Return type */
	typedef SGMatrix<T> ReturnType;

	/**Compute the cholesky decomposition \f$A = L L^{*}\f$ or \f$A = U^{*} U\f$ of a Hermitian positive definite matrix
	 * @param A - the matrix whose cholesky decomposition is to be computed
	 * @param lower - whether to compute the upper or lower triangular Cholesky factorization (default:lower)
	 * @return the upper or lower triangular Cholesky factorization
	 */
	static ReturnType compute(Matrix A, bool lower);

};

/**
 * @brief Partial specialization of add for the Eigen3 backend
 */
template <class Matrix>
struct cholesky<Backend::EIGEN3, Matrix>
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/** Return type */
	typedef SGMatrix<T> ReturnType;

	/** Eigen3 matrix type */
	typedef Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> MatrixXt;

	/**Compute the cholesky decomposition \f$A = L L^{*}\f$ or \f$A = U^{*} U\f$ of a Hermitian positive definite matrix
	 * @param A - the matrix whose cholesky decomposition is to be computed
	 * @param lower - whether to compute the upper or lower triangular Cholesky factorization (default:lower)
	 * @return the upper or lower triangular Cholesky factorization
	 */
	static ReturnType compute(SGMatrix<T> A, bool lower)
	{
		//creating eigen3 dense matrices
		Eigen::Map<MatrixXt> map_A(A.matrix, A.num_rows, A.num_cols);

		ReturnType cho(A.num_rows,A.num_cols);
		cho.set_const(0.0);
		Eigen::Map<MatrixXt> map_cho(cho.matrix, cho.num_rows, cho.num_cols);

		Eigen::LLT<MatrixXt> llt(map_A);

		//compute matrix L or U
		if(lower==false)
			map_cho= llt.matrixU();
		else
			map_cho= llt.matrixL();

		// checking for success
		REQUIRE(llt.info()!=Eigen::NumericalIssue, "Matrix is not Hermitian positive definite!\n");

		return cho;
	}

};

}

}

}
#endif //CHOLESKY_IMPL_H_
