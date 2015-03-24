/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2015 Yingrui Chang
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
