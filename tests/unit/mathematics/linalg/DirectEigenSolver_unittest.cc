/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Soumyajit De
 */

#include <shogun/lib/config.h>

#include <shogun/mathematics/linalg/linop/DenseMatrixOperator.h>
#include <shogun/mathematics/linalg/eigsolver/DirectEigenSolver.h>
#include <shogun/lib/SGMatrix.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(DirectEigenSolver, compute)
{
	const index_t size=2;
	SGMatrix<float64_t> m(size, size);
	m(0,0)=2.0;
	m(0,1)=1.0;
	m(1,0)=1.0;
	m(1,1)=3.0;

	CDenseMatrixOperator<float64_t>* A=new CDenseMatrixOperator<float64_t>(m);
	SG_REF(A);

	CDirectEigenSolver eig_solver(A);
	eig_solver.compute();

	float64_t min_eigval=eig_solver.get_min_eigenvalue();
	float64_t max_eigval=eig_solver.get_max_eigenvalue();

	EXPECT_NEAR(min_eigval, 1.38196601125010509747, 1E-15);
	EXPECT_NEAR(max_eigval, 3.61803398874989445844, 1E-15);

	SG_UNREF(A);
}
