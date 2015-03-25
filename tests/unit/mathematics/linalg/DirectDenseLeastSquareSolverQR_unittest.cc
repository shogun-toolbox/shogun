/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (W) 2015 Yingrui Chang
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
 */

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/linalg/linop/DenseMatrixOperator.h>
#include <shogun/mathematics/linalg/linsolver/DirectDenseLeastSquareSolverQR.h>
#include <shogun/lib/SGMatrix.h>
#include <gtest/gtest.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/Random.h>

using namespace shogun;

TEST(DirectDenseLeastSquareSolverQR, compute)
{
	//setup matix size and random seed
	const index_t rows = 10;
	const index_t cols = 20;
	const index_t randSeed = 0;

	//generate matrix m and dense operator A
	CRandom randGenerator(randSeed);
	SGMatrix<float64_t> m(rows, cols);
	for (index_t rowIndex=0; rowIndex<rows; ++rowIndex)
		for (index_t colIndex=0; colIndex<cols; ++colIndex)
			m(rowIndex, colIndex) = randGenerator.random(0.0, 10.0);

	CDenseMatrixOperator<float64_t> A(m);

	//setup RHS vector
	SGVector<float64_t> b(cols);
	for (index_t index=0; index<cols; ++index)
		b[index] = randGenerator.random(0.0, 10.0);

	//setup QR decomposition find the least square solution
	CDirectDenseLeastSquareSolverQR qr_solver;
	SGVector<float64_t> x = qr_solver.solve(&A,b);

	//check normal condition
	Eigen::Map<Eigen::MatrixXd> map_m(&m(0,0), rows, cols);
	Eigen::Map<Eigen::VectorXd> map_x(&x[0], rows);
	Eigen::Map<Eigen::VectorXd> map_b(&b[0], cols);

	EXPECT_NEAR((map_m*(map_b-map_m.transpose()*map_x)).norm(), 0.0, 1E-10);
}
#endif //HAVE_EIGEN3
