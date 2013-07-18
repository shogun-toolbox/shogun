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
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGSparseMatrix.h>
#include <shogun/features/SparseFeatures.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/logdet/SparseMatrixOperator.h>
#include <gtest/gtest.h>

using namespace shogun;
using namespace Eigen;

TEST(SparseMatrixOperator, symmetric_apply)
{
	const index_t size=2;

	SGMatrix<float64_t> m(size, size);
	m.set_const(0.0);
	m(0,0)=1.0;
	m(0,1)=0.01;
	m(1,1)=2.0;

	CSparseFeatures<float64_t> feat(m);
	SGSparseMatrix<float64_t> mat=feat.get_sparse_feature_matrix();
	CSparseMatrixOperator<float64_t, float64_t> op(mat);

	SGVector<float64_t> b(size);
	b.set_const(0.25);

	SGVector<float64_t> result=op.apply(b);
	Map<VectorXd> map_result(result.vector, result.vlen);

#ifdef EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
	EXPECT_NEAR(map_result.norm(), 0.56125417593101246, 1E-16);
#else
	const SparseMatrix<float64_t> &eig_m
		=EigenSparseUtil<float64_t>::toEigenSparse(mat);
	Map<VectorXd> map_b(b.vector, b.vlen);

	EXPECT_NEAR(map_result.norm(), (eig_m*map_b).norm(), 1E-16);
#endif
}

TEST(SparseMatrixOperator, symmetric_apply_complex_float)
{
	const index_t size=2;

	SGMatrix<complex64_t> m(size, size);
	m.set_const(complex64_t(0.0));
	m(0,0)=complex64_t(1.0);
	m(0,1)=complex64_t(0.01);
	m(1,1)=complex64_t(2.0);

	CSparseFeatures<complex64_t> feat(m);
	SGSparseMatrix<complex64_t> mat=feat.get_sparse_feature_matrix();
	CSparseMatrixOperator<complex64_t, float64_t> op(mat);

	SGVector<float64_t> b(size);
	b.set_const(0.25);

	SGVector<complex64_t> result=op.apply(b);
	Map<VectorXcd> map_result(result.vector, result.vlen);

#ifdef EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
	EXPECT_NEAR(map_result.norm(), 0.56125417593101246, 1E-16);
#else
	const SparseMatrix<complex64_t> &eig_m
		=EigenSparseUtil<complex64_t>::toEigenSparse(mat);
	Map<VectorXd> map_b(b.vector, b.vlen);

	EXPECT_NEAR(map_result.norm(), (eig_m*map_b.cast<complex64_t>()).norm(), 1E-16);
#endif
}

TEST(SparseMatrixOperator, asymmetric_apply)
{
	const index_t size=2;

	SGMatrix<float64_t> m(size*10, size);
	m.set_const(0.0);
	m(0,0)=-0.3435774457;
	m(9,0)=0.1253463474;
	m(19,0)=-2.34654537245;
	m(2,1)=1.23534643643;
	m(15,1)=-0.23462346332;
	m(17,1)=-1.12351352;

	CSparseFeatures<float64_t> feat(m);
	SGSparseMatrix<float64_t> mat=feat.get_sparse_feature_matrix();
	CSparseMatrixOperator<float64_t, float64_t> op(mat);

	SGVector<float64_t> b(size*10);
	b.set_const(0.25);

	SGVector<float64_t> result=op.apply(b);
	Map<VectorXd> map_result(result.vector, result.vlen);

#ifdef EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
	EXPECT_NEAR(map_result.norm(), 0.64192853298275987, 1E-16);
#else
	const SparseMatrix<float64_t> &eig_m
		=EigenSparseUtil<float64_t>::toEigenSparse(mat);
	Map<VectorXd> map_b(b.vector, b.vlen);

	EXPECT_NEAR(map_result.norm(), (eig_m*map_b).norm(), 1E-16);
#endif
}

TEST(SparseMatrixOperator, get_set_diagonal_no_alloc)
{
	const index_t size=2;

	SGMatrix<float64_t> m(size, size);
	m.set_const(0.0);
	m(0,0)=1.0;
	m(0,1)=0.01;
	m(1,1)=2.0;

	CSparseFeatures<float64_t> feat(m);
	SGSparseMatrix<float64_t> mat=feat.get_sparse_feature_matrix();
	CSparseMatrixOperator<float64_t, float64_t> op(mat);

	// get the old diagonal and check if it works fine
	SGVector<float64_t> old_diag=op.get_diagonal();
	Map<VectorXd> map_old_diag(old_diag.vector, old_diag.vlen);
	VectorXd eig_old_diag(size);
	eig_old_diag << 1.0, 2.0;

	EXPECT_NEAR(map_old_diag.norm(), eig_old_diag.norm(), 1E-16);

	// set the new diagonal and check if it works fine
	SGVector<float64_t> diag(size);
	diag.set_const(3.0);
	op.set_diagonal(diag);
	SGVector<float64_t> new_diag=op.get_diagonal();
	
	Map<VectorXd> map_diag(diag.vector, diag.vlen);
	Map<VectorXd> map_new_diag(new_diag.vector, new_diag.vlen);

	EXPECT_NEAR(map_diag.norm(), map_new_diag.norm(), 1E-16);
}

TEST(SparseMatrixOperator, get_set_diagonal_realloc)
{
	const index_t size=2;

	SGMatrix<float64_t> m(size, size);
	m.set_const(0.0);
	m(0,1)=0.01;
	m(1,1)=2.0;

	CSparseFeatures<float64_t> feat(m);
	SGSparseMatrix<float64_t> mat=feat.get_sparse_feature_matrix();
	CSparseMatrixOperator<float64_t, float64_t> op(mat);

	// get the old diagonal and check if it works fine
	SGVector<float64_t> old_diag=op.get_diagonal();
	Map<VectorXd> map_old_diag(old_diag.vector, old_diag.vlen);
	VectorXd eig_old_diag(size);
	eig_old_diag << 0.0, 2.0;

	EXPECT_NEAR(map_old_diag.norm(), eig_old_diag.norm(), 1E-16);

	// set the new diagonal and check if it works fine
	SGVector<float64_t> diag(size);
	diag.set_const(3.0);
	op.set_diagonal(diag);
	SGVector<float64_t> new_diag=op.get_diagonal();
	
	Map<VectorXd> map_diag(diag.vector, diag.vlen);
	Map<VectorXd> map_new_diag(new_diag.vector, new_diag.vlen);

	EXPECT_NEAR(map_diag.norm(), map_new_diag.norm(), 1E-16);
}

TEST(SparseMatrixOperator, get_set_diagonal_realloc_complex64)
{
	const index_t size=2;

	SGMatrix<complex64_t> m(size, size);
	m.set_const(complex64_t(0.0));
	m(0,1)=complex64_t(0.01);
	m(1,1)=complex64_t(2.0);

	CSparseFeatures<complex64_t> feat(m);
	SGSparseMatrix<complex64_t> mat=feat.get_sparse_feature_matrix();
	CSparseMatrixOperator<complex64_t, float64_t> op(mat);

	// get the old diagonal and check if it works fine
	SGVector<complex64_t> old_diag=op.get_diagonal();
	Map<VectorXcd> map_old_diag(old_diag.vector, old_diag.vlen);
	VectorXcd eig_old_diag(size);
	eig_old_diag(0)=complex64_t(0.0);
	eig_old_diag(1)=complex64_t(2.0);

	EXPECT_NEAR(map_old_diag.norm(), eig_old_diag.norm(), 1E-16);

	// set the new diagonal and check if it works fine
	SGVector<complex64_t> diag(size);
	diag.set_const(complex64_t(3.0));
	op.set_diagonal(diag);
	SGVector<complex64_t> new_diag=op.get_diagonal();
	
	Map<VectorXcd> map_diag(diag.vector, diag.vlen);
	Map<VectorXcd> map_new_diag(new_diag.vector, new_diag.vlen);

	EXPECT_NEAR(map_diag.norm(), map_new_diag.norm(), 1E-16);
}
#endif // HAVE_EIGEN3
