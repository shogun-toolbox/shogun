/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Wu Lin
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
 *
 * Code adapted from
 * Gaussian Process Machine Learning Toolbox
 * http://www.gaussianprocess.org/gpml/code/matlab/doc/
 */

#include <shogun/machine/gp/MatrixOperations.h>

#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/eigen3.h>

using namespace Eigen;

namespace shogun
{

SGMatrix<float64_t> CMatrixOperations::get_choleksy(SGVector<float64_t> W,
	SGVector<float64_t> sW, SGMatrix<float64_t> kernel, float64_t scale)
{
	Map<VectorXd> eigen_W(W.vector, W.vlen);
	Map<VectorXd> eigen_sW(sW.vector, sW.vlen);
	Map<MatrixXd> eigen_kernel(kernel.matrix, kernel.num_rows, kernel.num_cols);

	REQUIRE(eigen_W.rows()==eigen_sW.rows(),
		"The length of W (%d) and sW (%d) should be the same\n",
		eigen_W.rows(), eigen_sW.rows());
	REQUIRE(eigen_kernel.rows()==eigen_kernel.cols(),
		"Kernel should a sqaure matrix, row (%d) col (%d)\n",
		eigen_kernel.rows(), eigen_kernel.cols());
	REQUIRE(eigen_kernel.rows()==eigen_W.rows(),
		"The dimension between kernel (%d) and W (%d) should be matched\n",
		eigen_kernel.rows(), eigen_W.rows());

	SGMatrix<float64_t> L(eigen_W.rows(), eigen_W.rows());

	Map<MatrixXd> eigen_L(L.matrix, L.num_rows, L.num_cols);

	if (eigen_W.minCoeff()<0)
	{
		// compute inverse of diagonal noise: iW = 1/W
		VectorXd eigen_iW=(VectorXd::Ones(eigen_W.rows())).cwiseQuotient(eigen_W);

		FullPivLU<MatrixXd> lu(
			eigen_kernel*CMath::sq(scale)+MatrixXd(eigen_iW.asDiagonal()));

		// compute cholesky: L = -(K + iW)^-1
		eigen_L=-lu.inverse();
	}
	else
	{
		// compute cholesky: L = chol(sW * sW' .* K + I)
		LLT<MatrixXd> llt(
			(eigen_sW*eigen_sW.transpose()).cwiseProduct(eigen_kernel*CMath::sq(scale))+
			MatrixXd::Identity(eigen_kernel.rows(), eigen_kernel.cols()));

		eigen_L=llt.matrixU();
	}

	return L;
}

SGMatrix<float64_t> CMatrixOperations::get_inverse(SGMatrix<float64_t> L, SGMatrix<float64_t> kernel,
	SGVector<float64_t> sW, float64_t scale)
{
	Map<MatrixXd> eigen_L(L.matrix, L.num_rows, L.num_cols);
	Map<MatrixXd> eigen_kernel(kernel.matrix, kernel.num_rows, kernel.num_cols);
	Map<VectorXd> eigen_sW(sW.vector, sW.vlen);
	SGMatrix<float64_t> V(L.num_rows, L.num_cols);
	Map<MatrixXd> eigen_V(V.matrix, V.num_rows, V.num_cols);

	// compute V = L^(-1) * W^(1/2) * K, using upper triangular factor L^T
	eigen_V=eigen_L.triangularView<Upper>().adjoint().solve(
		eigen_sW.asDiagonal()*eigen_kernel*CMath::sq(scale));

	return get_inverse(L, kernel, sW, V, scale);
}

SGMatrix<float64_t> CMatrixOperations::get_inverse(SGMatrix<float64_t> L,
	SGMatrix<float64_t> kernel, SGVector<float64_t> sW, SGMatrix<float64_t> V,
	float64_t scale)
{
	Map<MatrixXd> eigen_L(L.matrix, L.num_rows, L.num_cols);
	Map<MatrixXd> eigen_kernel(kernel.matrix, kernel.num_rows, kernel.num_cols);
	Map<VectorXd> eigen_sW(sW.vector, sW.vlen);
	Map<MatrixXd> eigen_V(V.matrix, V.num_rows, V.num_cols);

	REQUIRE(eigen_kernel.rows()==eigen_kernel.cols(),
		"Kernel should a sqaure matrix, row (%d) col (%d)\n",
		eigen_kernel.rows(), eigen_kernel.cols());
	REQUIRE(eigen_L.rows()==eigen_L.cols(),
		"L should a sqaure matrix, row (%d) col (%d)\n",
		eigen_L.rows(), eigen_L.cols());
	REQUIRE(eigen_kernel.rows()==eigen_sW.rows(),
		"The dimension between kernel (%d) and sW (%d) should be matched\n",
		eigen_kernel.rows(), eigen_sW.rows());
	REQUIRE(eigen_L.rows()==eigen_sW.rows(),
		"The dimension between L (%d) and sW (%d) should be matched\n",
		eigen_L.rows(), eigen_sW.rows());


	SGMatrix<float64_t> tmp(eigen_L.rows(), eigen_L.cols());
	Map<MatrixXd> eigen_tmp(tmp.matrix, tmp.num_rows, tmp.num_cols);

	// compute covariance matrix of the posterior:
	// Sigma = K - K * W^(1/2) * (L * L^T)^(-1) * W^(1/2) * K =
	// K - (K * W^(1/2)) * (L^T)^(-1) * L^(-1) * W^(1/2) * K =
	// K - (W^(1/2) * K)^T * (L^(-1))^T * L^(-1) * W^(1/2) * K = K - V^T * V
	eigen_tmp=eigen_kernel*CMath::sq(scale)-eigen_V.adjoint()*eigen_V;
	return tmp;
}

} /* namespace shogun */
