/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Pan Deng, Soumyajit De, Bjoern Esser, Viktor Gal
 */

#include <gtest/gtest.h>

#include <shogun/lib/common.h>

#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/features/SparseFeatures.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/linalg/linop/SparseMatrixOperator.h>
#include <shogun/mathematics/linalg/linsolver/ConjugateGradientSolver.h>

using namespace shogun;
using namespace Eigen;

TEST(ConjugateGradientSolver, solve)
{
	const int32_t size=5;
	SGMatrix<float64_t> m(size, size);
	m.set_const(0.0);
	for (index_t i=0; i<size; ++i)
		m(i,i)=(i+1)*10000;

	SparseFeatures<float64_t> feat(m);
	SGSparseMatrix<float64_t> mat=feat.get_sparse_feature_matrix();

	auto A
		=std::make_shared<SparseMatrixOperator<float64_t>>(mat);

	ConjugateGradientSolver linear_solver;

	SGVector<float64_t> b(size);
	b.set_const(0.01);

	SGVector<float64_t> x=linear_solver.solve(A, b);
	Map<VectorXd> map_x(x.vector, x.vlen);

	Map<MatrixXd> map_m(m.matrix, m.num_rows, m.num_cols);
	Map<VectorXd> map_b(b.vector, b.vlen);

	EXPECT_NEAR(Math::abs((map_x-map_m.llt().solve(map_b)).norm()), 0.0, 1E-5);


}
