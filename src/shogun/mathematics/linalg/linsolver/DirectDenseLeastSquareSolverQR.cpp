/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2015 Yingrui Chang
 */

#include <shogun/lib/config.h>
#include <iostream>

#ifdef HAVE_EIGEN3
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/linalg/linop/DenseMatrixOperator.h>
#include <shogun/mathematics/linalg/linsolver/DirectDenseLeastSquareSolverQR.h>
#include <Eigen/LU>
#include <shogun/mathematics/eigen3.h>

using namespace Eigen;

namespace shogun
{

CDirectDenseLeastSquareSolverQR::CDirectDenseLeastSquareSolverQR()
	: CLinearSolver<float64_t, float64_t>()
{
}

CDirectDenseLeastSquareSolverQR::~CDirectDenseLeastSquareSolverQR()
{
}

SGVector<float64_t> CDirectDenseLeastSquareSolverQR::solve(
		CLinearOperator<SGVector<float64_t>, SGVector<float64_t> >* A, SGVector<float64_t> b)
{
	REQUIRE(A, "Operator is NULL!\n");
	REQUIRE(A->get_dimension() == b.vlen, "Dimension mismatch!\n");
	CDenseMatrixOperator<float64_t>* op
		=dynamic_cast<CDenseMatrixOperator<float64_t>*>(A);
	REQUIRE(op, "Operator is not DenseMatrixOperator type!\n");

	// creating eigen3 Dense Matrix
	SGMatrix<float64_t> sm = op->get_matrix_operator();
	Map<MatrixXd> map_m(sm.matrix, sm.num_rows, sm.num_cols);

	// creating eigen3 maps for vectors
	SGVector<float64_t> x(sm.num_rows);
	x.set_const(0.0);
	Map<VectorXd> map_x(x.vector, x.vlen);
	Map<VectorXd> map_b(b.vector, b.vlen);
	
	// using Householder QR decomposition to find the least square solution to A^T x = b
	HouseholderQR<MatrixXd> hqr;
	hqr.compute(map_m.transpose());
	map_x = hqr.solve(map_b);

	return x;
}

}
#endif // HAVE_EIGEN3
