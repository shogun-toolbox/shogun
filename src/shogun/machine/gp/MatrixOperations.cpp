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
 * xxx
 * and the reference paper is
 * xxx
 */

#include <shogun/machine/gp/MatrixOperations.h>

#ifdef HAVE_EIGEN3
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>

using namespace Eigen;

namespace shogun
{

SGMatrix<float64_t> CMatrixOperations::get_choleksy(SGVector<float64_t> W,
	SGVector<float64_t> sW, SGMatrix<float64_t> kernel, float64_t scale)
{
	REQUIRE(W.vlen == sW.vlen, "the length of W and sW should be the same");
	REQUIRE(kernel.num_rows == kernel.num_cols, "kernel should a sqaure matrix");
	REQUIRE(kernel.num_rows == W.vlen, "the dimension between kernel and W should be matched");

	Map<VectorXd> eigen_W (W.vector, W.vlen);
	Map<VectorXd> eigen_sW (sW.vector, sW.vlen);
	Map<MatrixXd> eigen_Kernel (kernel.matrix, kernel.num_rows, kernel.num_cols);

	SGMatrix<float64_t> L(W.vlen, W.vlen);

	Map<MatrixXd> eigen_L(L.matrix, L.num_rows, L.num_cols);

	if (eigen_W.minCoeff() < 0)
	{
		// compute inverse of diagonal noise: iW = 1/W
		VectorXd eigen_iW = (VectorXd::Ones(eigen_W.rows())).cwiseQuotient(eigen_W);

		FullPivLU<MatrixXd> lu(
			eigen_Kernel*CMath::sq(scale)+MatrixXd(eigen_iW.asDiagonal()));

		// compute cholesky: L = -(K + iW)^-1
		eigen_L = -lu.inverse();
	}
	else
	{
		// compute cholesky: L = chol(sW * sW' .* K + I)
		LLT<MatrixXd> llt(
			(eigen_sW*eigen_sW.transpose()).cwiseProduct(eigen_Kernel*CMath::sq(scale))+
			MatrixXd::Identity(eigen_Kernel.rows(), eigen_Kernel.cols()));

		eigen_L = llt.matrixU();
	}

	return L;
}

SGMatrix<float64_t> CMatrixOperations::get_inverse(SGMatrix<float64_t> L,
	SGMatrix<float64_t> kernel, SGVector<float64_t> sW, float64_t scale)
{
	REQUIRE(kernel.num_rows == kernel.num_cols, "");
	REQUIRE(L.num_rows == L.num_cols, "");
	REQUIRE(sW.vlen == L.num_rows, "");
	REQUIRE(sW.vlen == kernel.num_rows, "");

	SGMatrix<float64_t> tmp(L.num_rows, L.num_cols);
	Map<MatrixXd> eigen_tmp(tmp.matrix, tmp.num_rows, tmp.num_cols);

	Map<MatrixXd> eigen_L(L.matrix, L.num_rows, L.num_cols);
	Map<MatrixXd> eigen_Kernel(kernel.matrix, kernel.num_rows, kernel.num_cols);
	Map<VectorXd> eigen_sW(sW.vector, sW.vlen);

	// compute V = L^(-1) * W^(1/2) * K, using upper triangular factor L^T
	MatrixXd eigen_V=eigen_L.triangularView<Upper>().adjoint().solve(
		eigen_sW.asDiagonal()*eigen_Kernel*CMath::sq(scale));

	// compute covariance matrix of the posterior:
	// Sigma = K - K * W^(1/2) * (L * L^T)^(-1) * W^(1/2) * K =
	// K - (K * W^(1/2)) * (L^T)^(-1) * L^(-1) * W^(1/2) * K =
	// K - (W^(1/2) * K)^T * (L^(-1))^T * L^(-1) * W^(1/2) * K = K - V^T * V
	eigen_tmp = eigen_Kernel*CMath::sq(scale)-eigen_V.adjoint()*eigen_V;
	return tmp;
}

float64_t CMatrixOperations::get_log_det(MatrixXd eigen_A)
{

	REQUIRE(eigen_A.rows() == eigen_A.cols(), "Input matrix should be a sqaure matrix");

	PartialPivLU<MatrixXd> lu(eigen_A);
	VectorXd tmp(eigen_A.rows());

	for(index_t idx = 0; idx < tmp.rows(); idx++)
		tmp[idx] = idx+1;

	VectorXd p = lu.permutationP() * tmp;
	int detP = 1;

	for(index_t idx = 0; idx < p.rows(); idx++)
	{
		if (p[idx] != idx+1)
		{
			detP *= -1;
			index_t j = idx + 1;
			while(j < p.rows())
			{
				if (p[j] == idx+1)
					break;
				j ++;
			}
			p[j] = p[idx];
		}
	}

	VectorXd u = static_cast<MatrixXd>(lu.matrixLU().triangularView<Upper>()).diagonal();
	int check_u = 1;

	for(int idx = 0; idx < u.rows(); idx++)
	{
		if (u[idx] < 0)
			check_u *= -1;
		else if (u[idx] == 0)
		{
			check_u = 0;
			break;
		}
	}

	float64_t result = CMath::INFTY;

	if (check_u == detP)
		result = u.array().abs().log().sum();

	return result;
}

float64_t CMatrixOperations::get_log_det(SGMatrix<float64_t> A)
{
	Map<MatrixXd> eigen_A(A.matrix, A.num_rows, A.num_cols);
	return get_log_det(eigen_A);
}

} /* namespace shogun */
#endif /* HAVE_EIGEN3 */
