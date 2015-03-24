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
#include <shogun/mathematics/linalg/linsolver/DirectDenseLinearSolverLLT.h>
#include <shogun/lib/SGMatrix.h>
#include <gtest/gtest.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/Random.h>

using namespace shogun;

TEST(DirectDenseLinearSolverLLT, compute)
{
	//setup matix size and random seed
	const index_t size = 10;
	const index_t randSeed = 0;

	//generate SPD matrix m and dense operator A
	CRandom randGenerator(randSeed);
	SGMatrix<float64_t> m(size, size);
	for (index_t rowIndex=0; rowIndex<size; ++rowIndex)
		for (index_t colIndex=0; colIndex<size; ++colIndex)
			m(rowIndex, colIndex) = randGenerator.random(0.0, 10.0);
	
	Eigen::Map<Eigen::MatrixXd> map_m(&m(0,0), size, size);
	map_m=map_m*map_m.transpose();

	CDenseMatrixOperator<float64_t> A(m);

	//setup RHS vector
	SGVector<float64_t> b(size);
	for (index_t index=0; index<size; ++index)
		b[index] = randGenerator.random(0.0, 10.0);

	//setup linear solver and solve the system
	CDirectDenseLinearSolverLLT linear_solver;
	SGVector<float64_t> x = linear_solver.solve(&A,b);

	//check result
	Eigen::Map<Eigen::VectorXd> map_x(&x[0], size);
	Eigen::Map<Eigen::VectorXd> map_b(&b[0], size);

	EXPECT_NEAR((map_m*map_x-map_b).norm(), 0.0, 1E-10);
}
#endif //HAVE_EIGEN3
