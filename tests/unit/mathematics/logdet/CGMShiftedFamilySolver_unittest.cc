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
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGSparseMatrix.h>
#include <shogun/features/SparseFeatures.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/logdet/SparseMatrixOperator.h>
#include <shogun/mathematics/logdet/DenseMatrixOperator.h>
#include <shogun/mathematics/logdet/ConjugateGradientSolver.h>
#include <shogun/mathematics/logdet/DirectLinearSolverComplex.h>
#include <shogun/mathematics/logdet/CGMShiftedFamilySolver.h>
#include <gtest/gtest.h>

using namespace shogun;
using namespace Eigen;

TEST(CGMShiftedFamilySolver, solve_shifted_weight_noshift)
{
	const int32_t size=10;
	SGMatrix<float64_t> m(size, size);
	m.set_const(0.0);

	// diagonal Hermintian matrix
	for (index_t i=0; i<size; ++i)
		m(i,i)=i+1;

	// constant vector of the system
	SGVector<float64_t> b(size);
	b.set_const(0.5);

	// Creating sparse system to solve with CG_M
	CSparseFeatures<float64_t> feat(m);
	SGSparseMatrix<float64_t> mat=feat.get_sparse_feature_matrix();
	CSparseMatrixOperator<float64_t>* A
		=new CSparseMatrixOperator<float64_t>(mat);

	// Solve with CG_M
	CCGMShiftedFamilySolver cg_m_linear_solver;
	SGVector<float64_t> x_sh=cg_m_linear_solver.solve(A, b);

	// checking with plain CG solver since shift is zero
	CConjugateGradientSolver cg_linear_solver;
	SGVector<float64_t> x=cg_linear_solver.solve(A, b);

	Map<VectorXd> x_sh_map(x_sh.vector, x_sh.vlen);
	Map<VectorXd> x_map(x.vector, x.vlen);

	EXPECT_NEAR((x_sh_map-x_map).norm(), 0.0, 1E-14);

	SG_UNREF(A);
}

TEST(CGMShiftedFamilySolver, solve_shifted_weight_real_shift)
{
	const int32_t size=10;
	SGMatrix<float64_t> m(size, size);
	m.set_const(0.0);

	// diagonal Hermintian matrix
	for (index_t i=0; i<size; ++i)
		m(i,i)=i+1;

	// constant vector of the system
	SGVector<float64_t> b(size);
	b.set_const(0.5);

	// shifts
	float64_t shift=0.01;

	SGVector<complex64_t> shifts(1);
	shifts.set_const(shift);

	// weights
	SGVector<complex64_t> weights(1);
	weights.set_const(1.0);
	
	// Creating sparse system to solve with CG_M
	CSparseFeatures<float64_t> feat(m);
	SGSparseMatrix<float64_t> mat=feat.get_sparse_feature_matrix();
	CSparseMatrixOperator<float64_t>* A
		=new CSparseMatrixOperator<float64_t>(mat);

	// Solve with CG_M
	CCGMShiftedFamilySolver cg_m_linear_solver;
	SGVector<complex64_t> x_sh
		=cg_m_linear_solver.solve_shifted_weighted(A, b, shifts, weights);

	// checking with plain CG solver since number of shifts is 1
	for (index_t i=0; i<size; ++i)
		m(i,i)=m(i,i)+shift;

	CSparseFeatures<float64_t> feat2(m);
	mat=feat2.get_sparse_feature_matrix();
	SG_UNREF(A);
	A=new CSparseMatrixOperator<float64_t>(mat);

	CConjugateGradientSolver cg_linear_solver;
	SGVector<float64_t> x=cg_linear_solver.solve(A, b);

	Map<VectorXcd> x_sh_map(x_sh.vector, x_sh.vlen);
	Map<VectorXd> x_map(x.vector, x.vlen);

	EXPECT_NEAR((x_sh_map-x_map.cast<complex64_t>()).norm(), 0.0, 0.01);

	SG_UNREF(A);
}

TEST(CGMShiftedFamilySolver, solve_shifted_weight_complex_shift)
{
	const int32_t size=10;
	SGMatrix<float64_t> m(size, size);
	m.set_const(0.0);

	// diagonal Hermintian matrix
	for (index_t i=0; i<size; ++i)
		m(i,i)=i+1;

	// constant vector of the system
	SGVector<float64_t> b(size);
	b.set_const(0.5);

	// shifts
	complex64_t shift(0.0, 0.01);

	SGVector<complex64_t> shifts(1);
	shifts.set_const(shift);

	// weights
	SGVector<complex64_t> weights(1);
	weights.set_const(1.0);
	
	// Creating sparse system to solve with CG_M
	CSparseFeatures<float64_t> feat(m);
	SGSparseMatrix<float64_t> mat=feat.get_sparse_feature_matrix();
	CSparseMatrixOperator<float64_t>* A
		=new CSparseMatrixOperator<float64_t>(mat);

	// Solve with CG_M
	CCGMShiftedFamilySolver cg_m_linear_solver;
	cg_m_linear_solver.set_iteration_limit(10000);
	SGVector<complex64_t> x_sh
		=cg_m_linear_solver.solve_shifted_weighted(A, b, shifts, weights);

	// checking with triangular solver since number of shifts is 1
	SGMatrix<complex64_t> m2(size, size);
	m2.set_const(0.0);
	for (index_t i=0; i<size; ++i)
		m2(i,i)=m(i,i)+shift;

	CDenseMatrixOperator<complex64_t>* B
		=new CDenseMatrixOperator<complex64_t>(m2);

	CDirectLinearSolverComplex direct_solver;
	SGVector<complex64_t> x=direct_solver.solve(B, b);

	Map<VectorXcd> x_sh_map(x_sh.vector, x_sh.vlen);
	Map<VectorXcd> x_map(x.vector, x.vlen);

	EXPECT_NEAR((x_sh_map-x_map).norm(), 0.0, 0.13);

	SG_UNREF(A);
	SG_UNREF(B);
}
#endif //HAVE_EIGEN3
