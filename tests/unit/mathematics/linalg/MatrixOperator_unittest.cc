/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soumyajit De, Pan Deng, Bjoern Esser, Viktor Gal
 */
#include <gtest/gtest.h>

#include <shogun/lib/common.h>

#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGSparseMatrix.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/linalg/linop/SparseMatrixOperator.h>
#include <shogun/mathematics/linalg/linop/DenseMatrixOperator.h>
#include <shogun/features/SparseFeatures.h>

using namespace shogun;
using namespace Eigen;

TEST(MatrixOperator, cast_dense_double_complex)
{
	const int32_t size=2;
	SGMatrix<float64_t> m(size, size);
	m.set_const(1.0);

	auto orig_op
		=std::make_shared<DenseMatrixOperator<float64_t>>(m);
	auto casted_op
		=std::make_shared<DenseMatrixOperator<complex128_t>>(*static_cast<DenseMatrixOperator<complex128_t>*>(*orig_op));

	SGMatrix<float64_t> orig_m=orig_op->get_matrix_operator();
	SGMatrix<complex128_t> casted_m=casted_op->get_matrix_operator();

	Map<MatrixXd> eig_orig(orig_m.matrix, orig_m.num_rows, orig_m.num_cols);
	Map<MatrixXcd> eig_casted(casted_m.matrix, casted_m.num_rows, casted_m.num_cols);

	EXPECT_NEAR((eig_orig.cast<complex128_t>()-eig_casted).norm(), 0.0, 1E-15);
}

TEST(MatrixOperator, cast_sparse_double_complex)
{
	const int32_t size=4;
	SGMatrix<float64_t> m(size, size);
	m.set_const(0.0);

	for (index_t i=0; i<size; ++i)
		m(i,i)=1.0;

	SparseFeatures<float64_t> feat(m);
	SGSparseMatrix<float64_t> sm=feat.get_sparse_feature_matrix();

	SparseMatrixOperator<float64_t>* orig_op
		=new SparseMatrixOperator<float64_t>(sm);
	SparseMatrixOperator<complex128_t>* casted_op
		=static_cast<SparseMatrixOperator<complex128_t>*>(*orig_op);

	SGSparseMatrix<complex128_t> casted_m=casted_op->get_matrix_operator();
	const SparseMatrix<complex128_t>& eig_casted
		=EigenSparseUtil<complex128_t>::toEigenSparse(casted_m);

	Map<MatrixXd> eig_orig(m.matrix, m.num_rows, m.num_cols);

	EXPECT_NEAR(2.0, (eig_orig*eig_casted).norm(), 1E-15);
	delete orig_op;
	delete casted_op;
}
