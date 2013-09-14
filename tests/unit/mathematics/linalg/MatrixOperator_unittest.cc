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
#include <shogun/lib/SGSparseMatrix.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/linalg/linop/SparseMatrixOperator.h>
#include <shogun/mathematics/linalg/linop/DenseMatrixOperator.h>
#include <shogun/features/SparseFeatures.h>
#include <gtest/gtest.h>

using namespace shogun;
using namespace Eigen;

TEST(MatrixOperator, cast_dense_double_complex)
{
	const int32_t size=2;
	SGMatrix<float64_t> m(size, size);
	m.set_const(1.0);

	CDenseMatrixOperator<float64_t>* orig_op
		=new CDenseMatrixOperator<float64_t>(m);
	CDenseMatrixOperator<complex64_t>* casted_op
		=static_cast<CDenseMatrixOperator<complex64_t>*>(*orig_op);

	SGMatrix<float64_t> orig_m=orig_op->get_matrix_operator();
	SGMatrix<complex64_t> casted_m=casted_op->get_matrix_operator();

	Map<MatrixXd> eig_orig(orig_m.matrix, orig_m.num_rows, orig_m.num_cols);
	Map<MatrixXcd> eig_casted(casted_m.matrix, casted_m.num_rows, casted_m.num_cols);
	
	EXPECT_NEAR((eig_orig.cast<complex64_t>()-eig_casted).norm(), 0.0, 1E-15);

	SG_UNREF(orig_op);
	SG_UNREF(casted_op);
}

TEST(MatrixOperator, cast_sparse_double_complex)
{
	const int32_t size=4;
	SGMatrix<float64_t> m(size, size);
	m.set_const(0.0);

	for (index_t i=0; i<size; ++i)
		m(i,i)=1.0;

	CSparseFeatures<float64_t> feat(m);
	SGSparseMatrix<float64_t> sm=feat.get_sparse_feature_matrix();

	CSparseMatrixOperator<float64_t>* orig_op
		=new CSparseMatrixOperator<float64_t>(sm);
	CSparseMatrixOperator<complex64_t>* casted_op
		=static_cast<CSparseMatrixOperator<complex64_t>*>(*orig_op);

	SGSparseMatrix<complex64_t> casted_m=casted_op->get_matrix_operator();
	const SparseMatrix<complex64_t>& eig_casted
		=EigenSparseUtil<complex64_t>::toEigenSparse(casted_m);

	Map<MatrixXd> eig_orig(m.matrix, m.num_rows, m.num_cols);

	EXPECT_NEAR((eig_orig*eig_casted).norm(), 2.0, 1E-15);

	SG_UNREF(orig_op);
	SG_UNREF(casted_op);
}
#endif // HAVE_EIGEN3
