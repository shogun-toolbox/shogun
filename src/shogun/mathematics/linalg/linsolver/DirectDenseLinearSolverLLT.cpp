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
#include <shogun/mathematics/eigen3.h>

#ifdef HAVE_EIGEN3
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/linalg/linop/DenseMatrixOperator.h>
#include <shogun/mathematics/linalg/linsolver/DirectDenseLinearSolverLLT.h>
#include <Eigen/LU>

using namespace Eigen;
using namespace std;

namespace shogun
{

CDirectDenseLinearSolverLLT::CDirectDenseLinearSolverLLT()
	: CLinearSolver<float64_t, float64_t>()
{
}

CDirectDenseLinearSolverLLT::~CDirectDenseLinearSolverLLT()
{
}

SGVector<float64_t> CDirectDenseLinearSolverLLT::solve(
		CLinearOperator<SGVector<float64_t>, SGVector<float64_t> >* A, SGVector<float64_t> b)
{
	REQUIRE(A, "Operator is NULL!\n");
	REQUIRE(A->get_dimension()==b.vlen, "Dimension mismatch!\n");
	CDenseMatrixOperator<float64_t>* op
		=dynamic_cast<CDenseMatrixOperator<float64_t>*>(A);
	REQUIRE(op, "Operator is not DenseMatrixOperator type!\n");

	// creating eigen3 Dense Matrix
	SGMatrix<float64_t> sm=op->get_matrix_operator();
	Map<MatrixXd> map_m(sm.matrix, sm.num_rows, sm.num_cols);

	// creating eigen3 maps for vectors
	SGVector<float64_t> x(sm.num_rows);
	x.set_const(0.0);
	Map<VectorXd> map_x(x.vector, x.vlen);
	Map<VectorXd> map_b(b.vector, b.vlen);
	
	// using LLT to solve the system Ax=b
	LLT<MatrixXd> llt;
	llt.compute(map_m);
	map_x=llt.solve(map_b);
	
	// checking for success
	if (llt.info()!=Success)
		SG_WARNING("LLU solver failed! maybe input operator is not symmetric positive definite.\n");

	return x;
}

}
#endif // HAVE_EIGEN3
