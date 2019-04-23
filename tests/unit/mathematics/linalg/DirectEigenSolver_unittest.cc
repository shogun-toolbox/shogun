/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Thoralf Klein, Pan Deng, Soumyajit De, Bjoern Esser, Viktor Gal
 */

#include <gtest/gtest.h>

#include <shogun/lib/config.h>

#include <shogun/mathematics/linalg/linop/DenseMatrixOperator.h>
#include <shogun/mathematics/linalg/eigsolver/DirectEigenSolver.h>
#include <shogun/lib/SGMatrix.h>

using namespace shogun;

TEST(DirectEigenSolver, compute)
{
	const index_t size=2;
	SGMatrix<float64_t> m(size, size);
	m(0,0)=2.0;
	m(0,1)=1.0;
	m(1,0)=1.0;
	m(1,1)=3.0;

	auto A=std::make_shared<DenseMatrixOperator<float64_t>>(m);


	DirectEigenSolver eig_solver(A);
	eig_solver.compute();

	float64_t min_eigval=eig_solver.get_min_eigenvalue();
	float64_t max_eigval=eig_solver.get_max_eigenvalue();

	EXPECT_NEAR(min_eigval, 1.38196601125010509747, 1E-15);
	EXPECT_NEAR(max_eigval, 3.61803398874989445844, 1E-15);


}
