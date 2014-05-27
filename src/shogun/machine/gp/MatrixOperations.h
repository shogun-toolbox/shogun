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

#ifndef CMATRIXOPERATIONS_H
#define CMATRIXOPERATIONS_H

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>
#include <iostream>

using namespace Eigen;

namespace shogun
{
/** @brief The helper class is used for Laplace and KL methods
 *
 *
 */

class CMatrixOperations
{
public:
	static MatrixXd get_choleksy(VectorXd eigen_W, VectorXd eigen_sW,
		MatrixXd eigen_Kernel, float64_t scale)
	{
		MatrixXd eigen_L;

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
			LLT<MatrixXd> L(
				(eigen_sW*eigen_sW.transpose()).cwiseProduct(eigen_Kernel*CMath::sq(scale))+
				MatrixXd::Identity(eigen_Kernel.rows(), eigen_Kernel.cols()));

			eigen_L = L.matrixU();
		}

		return eigen_L;
	}

	static MatrixXd get_inverse(MatrixXd eigen_L, MatrixXd eigen_Kernel,
		VectorXd eigen_sW, float64_t scale)
	{
		// compute V = L^(-1) * W^(1/2) * K, using upper triangular factor L^T
		MatrixXd eigen_V=eigen_L.triangularView<Upper>().adjoint().solve(
			eigen_sW.asDiagonal()*eigen_Kernel*CMath::sq(scale));

		// compute covariance matrix of the posterior:
		// Sigma = K - K * W^(1/2) * (L * L^T)^(-1) * W^(1/2) * K =
		// K - (K * W^(1/2)) * (L^T)^(-1) * L^(-1) * W^(1/2) * K =
		// K - (W^(1/2) * K)^T * (L^(-1))^T * L^(-1) * W^(1/2) * K = K - V^T * V
		return eigen_Kernel*CMath::sq(scale)-eigen_V.adjoint()*eigen_V;
	}

	static float64_t get_log_det(MatrixXd A)
	{

		REQUIRE(A.rows() == A.cols(), "Input matrix should be a sqaure matrix");

		PartialPivLU<MatrixXd> lu(A);
		VectorXd tmp(A.rows());
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
};
}
#endif /* HAVE_EIGEN3 */
#endif /* CMATRIXOPERATIONS_H */
