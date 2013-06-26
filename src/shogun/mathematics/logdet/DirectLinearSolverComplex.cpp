/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General turalPublic License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Written (W) 2013 Soumyajit De
 */

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/logdet/DenseMatrixOperator.h>
#include <shogun/mathematics/logdet/DirectLinearSolverComplex.h>

using namespace Eigen;

namespace shogun
{

CDirectLinearSolverComplex::CDirectLinearSolverComplex()
	: CLinearSolver<complex64_t>(),
	  m_type(DS_QR_NOPERM)
{
	SG_GCDEBUG("%s created (%p)\n", this->get_name(), this)
}

CDirectLinearSolverComplex::CDirectLinearSolverComplex(EDirectSolverType type)
	: CLinearSolver<complex64_t>(),
	  m_type(type)
{
	SG_GCDEBUG("%s created (%p)\n", this->get_name(), this)
}

CDirectLinearSolverComplex::~CDirectLinearSolverComplex()
{
	SG_GCDEBUG("%s destroyed (%p)\n", this->get_name(), this)
}

SGVector<complex64_t> CDirectLinearSolverComplex::solve(
		CLinearOperator<complex64_t>* A, SGVector<complex64_t> b)
{
	SG_DEBUG("Entering..\n");

	SGVector<complex64_t> x(b.vlen);

	REQUIRE(A, "Operator is NULL!\n");
	REQUIRE(A->get_dim()==b.vlen, "Dimension mismatch!\n");

	CDenseMatrixOperator<complex64_t> *op=
		dynamic_cast<CDenseMatrixOperator<complex64_t>*>(A);
	REQUIRE(op, "Operator is not CDenseMatrixOperator<complex64_t> type!\n");

	SGMatrix<complex64_t> mat_A=op->get_matrix_operator();
	Map<MatrixXcd> map_A(mat_A.matrix, mat_A.num_rows, mat_A.num_cols);
	Map<VectorXcd> map_b(b.vector, b.vlen);
	Map<VectorXcd> map_x(x.vector, x.vlen);

	// rank checking for LLT
	FullPivLU<MatrixXcd> lu(map_A);
	bool full_rank=lu.rank()==mat_A.num_cols;

	switch (m_type)
	{
	case DS_LLT:
		if (full_rank)
			map_x=map_A.llt().solve(map_b);
		else
			SG_WARNING("Matrix is not full rank!\n");
		break;
	case DS_QR_NOPERM:
		map_x=map_A.householderQr().solve(map_b);
		break;
	case DS_QR_COLPERM:
		map_x=map_A.colPivHouseholderQr().solve(map_b);
		break;
	case DS_QR_FULLPERM:
		map_x=map_A.fullPivHouseholderQr().solve(map_b);
		break;
	case DS_SVD:
		map_x=map_A.jacobiSvd(ComputeThinU|ComputeThinV).solve(map_b);
		break;
	};

	SG_DEBUG("Leaving..\n");
	return x;
}

}
#endif // HAVE_EIGEN3
