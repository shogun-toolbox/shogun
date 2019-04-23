/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soumyajit De, Pan Deng, Bjoern Esser, Viktor Gal
 */

#include <gtest/gtest.h>

#include <shogun/lib/common.h>

#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGSparseMatrix.h>
#include <shogun/features/SparseFeatures.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/linalg/linop/SparseMatrixOperator.h>
#include <shogun/mathematics/linalg/linop/DenseMatrixOperator.h>
#include <shogun/mathematics/linalg/linsolver/ConjugateGradientSolver.h>
#include <shogun/mathematics/linalg/linsolver/DirectLinearSolverComplex.h>
#include <shogun/mathematics/linalg/linsolver/CGMShiftedFamilySolver.h>

using namespace shogun;
using namespace Eigen;

TEST(CGMShiftedFamilySolver, solve_shifted_weight_noshift)
{
	const int32_t size=10;
	SGMatrix<float64_t> m(size, size);
	m.set_const(0.0);

	// diagonal Hermintian matrix
	for (index_t i=0; i<size; ++i)
		m(i,i)=Math::pow(2, i);

	// constant vector of the system
	SGVector<float64_t> b(size);
	b.set_const(0.5);

	// Creating sparse system to solve with CG_M
	SparseFeatures<float64_t> feat(m);
	SGSparseMatrix<float64_t> mat=feat.get_sparse_feature_matrix();
	auto A
		=std::make_shared<SparseMatrixOperator<float64_t>>(mat);

	// Solve with CG_M
	CGMShiftedFamilySolver cg_m_linear_solver;
	SGVector<float64_t> x_sh=cg_m_linear_solver.solve(A, b);

	// checking with plain CG solver since shift is zero
	ConjugateGradientSolver cg_linear_solver;
	SGVector<float64_t> x=cg_linear_solver.solve(A, b);

	Map<VectorXd> x_sh_map(x_sh.vector, x_sh.vlen);
	Map<VectorXd> x_map(x.vector, x.vlen);

	EXPECT_NEAR((x_sh_map-x_map).norm(), 0.0, 1E-15);


}

TEST(CGMShiftedFamilySolver, solve_shifted_weight_real_shift)
{
	const int32_t size=10;
	SGMatrix<float64_t> m(size, size);
	m.set_const(0.0);

	// diagonal Hermintian matrix
	for (index_t i=0; i<size; ++i)
		m(i,i)=Math::pow(2, i);

	// constant vector of the system
	SGVector<float64_t> b(size);
	b.set_const(1.0);

	// shifts
	float64_t shift=100;

	SGVector<complex128_t> shifts(1);
	shifts.set_const(shift);

	// weights
	SGVector<complex128_t> weights(1);
	weights.set_const(1.0);

	// Creating sparse system to solve with CG_M
	SparseFeatures<float64_t> feat(m);
	SGSparseMatrix<float64_t> mat=feat.get_sparse_feature_matrix();
	auto A
		=std::make_shared<SparseMatrixOperator<float64_t>>(mat);

	// Solve with CG_M
	CGMShiftedFamilySolver cg_m_linear_solver;
	SGVector<complex128_t> x_sh
		=cg_m_linear_solver.solve_shifted_weighted(A, b, shifts, weights);

	// checking with plain CG solver since number of shifts is 1
	for (index_t i=0; i<size; ++i)
		m(i,i)=m(i,i)+shift;

	SparseFeatures<float64_t> feat2(m);
	mat=feat2.get_sparse_feature_matrix();

	A=std::make_shared<SparseMatrixOperator<float64_t>>(mat);

	ConjugateGradientSolver cg_linear_solver;
	SGVector<float64_t> x=cg_linear_solver.solve(A, b);

	Map<VectorXcd> x_sh_map(x_sh.vector, x_sh.vlen);
	Map<VectorXd> x_map(x.vector, x.vlen);

	EXPECT_NEAR((x_sh_map-x_map.cast<complex128_t>()).norm(), 0.0, 1E-7);


}

TEST(CGMShiftedFamilySolver, solve_shifted_weight_complex_shift)
{
	const int32_t size=10;
	SGMatrix<float64_t> m(size, size);
	m.set_const(0.0);

	// diagonal Hermintian matrix
	for (index_t i=0; i<size; ++i)
		m(i,i)=Math::pow(2, i);

	// constant vector of the system
	SGVector<float64_t> b(size);
	b.set_const(0.5);

	// shifts
	complex128_t shift(0.0, 100.0);

	SGVector<complex128_t> shifts(1);
	shifts.set_const(shift);

	// weights
	SGVector<complex128_t> weights(1);
	weights.set_const(1.0);

	// Creating sparse system to solve with CG_M
	SparseFeatures<float64_t> feat(m);
	SGSparseMatrix<float64_t> mat=feat.get_sparse_feature_matrix();
	auto A
		=std::make_shared<SparseMatrixOperator<float64_t>>(mat);

	// Solve with CG_M
	CGMShiftedFamilySolver cg_m_linear_solver;
	SGVector<complex128_t> x_sh
		=cg_m_linear_solver.solve_shifted_weighted(A, b, shifts, weights);

	// checking with triangular solver since number of shifts is 1
	SGMatrix<complex128_t> m2(size, size);
	m2.set_const(0.0);
	for (index_t i=0; i<size; ++i)
		m2(i,i)=m(i,i)+shift;

	auto B
		=std::make_shared<DenseMatrixOperator<complex128_t>>(m2);

	DirectLinearSolverComplex direct_solver;
	SGVector<complex128_t> x=direct_solver.solve(B, b);

	Map<VectorXcd> x_sh_map(x_sh.vector, x_sh.vlen);
	Map<VectorXcd> x_map(x.vector, x.vlen);

	EXPECT_NEAR((x_sh_map-x_map).norm(), 0.0, 1E-15);



}
