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
#include <shogun/mathematics/logdet/SparseMatrixOperator.h>
#include <shogun/mathematics/logdet/DenseMatrixOperator.h>
#include <shogun/mathematics/logdet/DirectEigenSolver.h>
#include <shogun/mathematics/logdet/LanczosEigenSolver.h>
#include <gtest/gtest.h>

using namespace shogun;

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
#endif // HAVE_EIGEN3
#endif // HAVE_LAPACK
