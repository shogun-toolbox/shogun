/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Soumyajit De
 */
 
#include <shogun/lib/common.h>

#ifdef HAVE_EIGEN3
#ifdef HAVE_LAPACK

#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGSparseMatrix.h>
#include <shogun/features/SparseFeatures.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/logdet/SparseMatrixOperator.h>
#include <shogun/mathematics/logdet/DenseMatrixOperator.h>
#include <shogun/mathematics/logdet/DirectEigenSolver.h>
#include <shogun/mathematics/logdet/LanczosEigenSolver.h>
#include <gtest/gtest.h>

using namespace shogun;
using namespace Eigen;

TEST(LanczosEigenSolver, compute)
{
	const int32_t size=4;
	SGMatrix<float64_t> m(size, size);
	m.set_const(CMath::random(50.0, 100.0));
	
	// Hermintian matrix
	for (index_t i=0; i<size; ++i)
		m(i,i)=CMath::random(100.0, 10000.0);

	// Creating sparse linear operator to use with Lanczos
	CSparseFeatures<float64_t> feat(m);
	SGSparseMatrix<float64_t> mat=feat.get_sparse_feature_matrix();
	CEigenSolver* eig_solver=NULL;

	CLinearOperator<float64_t>* A=new CSparseMatrixOperator<float64_t>(mat);
	SG_REF(A);

	eig_solver=new CLanczosEigenSolver(A);
	eig_solver->compute();

	float64_t lanc_max_eig=eig_solver->get_max_eigenvalue();
	float64_t lanc_min_eig=eig_solver->get_min_eigenvalue();

	SG_UNREF(eig_solver);
	SG_UNREF(A);

	// create dense linear operator to use with direct eigensolver
	CDenseMatrixOperator<float64_t>* B=new CDenseMatrixOperator<float64_t>(m);
	SG_REF(B);

	eig_solver=new CDirectEigenSolver(B);
	eig_solver->compute();

	float64_t dir_max_eig=eig_solver->get_max_eigenvalue();
	float64_t dir_min_eig=eig_solver->get_min_eigenvalue();

	SG_UNREF(eig_solver);
	SG_UNREF(B);

	// compare these two
	EXPECT_NEAR(CMath::abs(lanc_max_eig-dir_max_eig), 0.0, 1E-6);
	EXPECT_NEAR(CMath::abs(lanc_min_eig-dir_min_eig), 0.0, 1E-6);
}

TEST(LanczosEigenSolver, compute_big_diag_matrix)
{
	float64_t difficulty=4;
	float64_t min_eigenvalue=0.0001;

	// create a sparse matrix	
	const index_t size=10000;
	SGSparseMatrix<float64_t> sm(size, size);
	CSparseMatrixOperator<float64_t>* op=new CSparseMatrixOperator<float64_t>(sm);
	SG_REF(op);

	// set its diagonal
	SGVector<float64_t> diag(size);
	for (index_t i=0; i<size; ++i)
	{
		diag[i]=CMath::pow(CMath::abs(sg_rand->std_normal_distrib()), difficulty)
			+min_eigenvalue;
	}
	op->set_diagonal(diag);

	CLanczosEigenSolver* eig_solver=new CLanczosEigenSolver(op);
	SG_REF(eig_solver);
	eig_solver->compute();

	// test eigenvalues
	Map<VectorXd> diag_map(diag.vector, diag.vlen);
	float64_t actual_min_eig=diag_map.minCoeff();
	float64_t actual_max_eig=diag_map.maxCoeff();
	float64_t computed_min_eig=eig_solver->get_min_eigenvalue();
	float64_t computed_max_eig=eig_solver->get_max_eigenvalue();
	EXPECT_NEAR(actual_min_eig, computed_min_eig, 1E-4);
	EXPECT_NEAR(actual_max_eig, computed_max_eig, 1E-4);

	SG_SPRINT("%.20lf, %.20lf\n", actual_min_eig, actual_max_eig);

	SG_UNREF(eig_solver);
	SG_UNREF(op);
}
#endif // HAVE_EIGEN3
#endif // HAVE_LAPACK
