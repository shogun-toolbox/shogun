/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soumyajit De, Sunil Mahendrakar, Pan Deng, Bjoern Esser, Viktor Gal
 */

#include <gtest/gtest.h>

#include <shogun/lib/config.h>

#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/linalg/linop/DenseMatrixOperator.h>
#include <shogun/mathematics/linalg/linsolver/DirectLinearSolverComplex.h>

using namespace shogun;
using namespace Eigen;

TEST(DirectLinearSolverComplex, solve_SVD)
{
	const index_t size=2;
	SGMatrix<complex128_t> m(size, size);
	m(0,0)=complex128_t(2.0);
	m(0,1)=complex128_t(1.0, 2.0);
	m(1,0)=complex128_t(1.0, 2.0);
	m(1,1)=complex128_t(3.0);

	auto A
		=std::make_shared<DenseMatrixOperator<complex128_t>>(m);

	SGVector<float64_t> b(size);
	b.set_const(1.0);

	DirectLinearSolverComplex solver(DS_SVD);
	SGVector<complex128_t> x
		=solver.solve(A->as<LinearOperator<complex128_t>>(), b);

	SGVector<complex128_t> bp=A->apply(x);
	Map<VectorXd> map_b(b.vector, b.vlen);
	Map<VectorXcd> map_bp(bp.vector, bp.vlen);

	EXPECT_NEAR((map_b.cast<complex128_t>()-map_bp).norm()/map_b.norm(),
		0.0, 1E-15);


}

TEST(DirectLinearSolverComplex, solve_QR_NOPERM)
{
	const index_t size=2;
	SGMatrix<complex128_t> m(size, size);
	m(0,0)=complex128_t(2.0);
	m(0,1)=complex128_t(1.0, 2.0);
	m(1,0)=complex128_t(1.0, 2.0);
	m(1,1)=complex128_t(3.0);

	auto A
		=std::make_shared<DenseMatrixOperator<complex128_t>>(m);

	SGVector<float64_t> b(size);
	b.set_const(1.0);

	DirectLinearSolverComplex solver(DS_QR_NOPERM);
	SGVector<complex128_t> x
		=solver.solve(A->as<LinearOperator<complex128_t>>(), b);

	SGVector<complex128_t> bp=A->apply(x);
	Map<VectorXd> map_b(b.vector, b.vlen);
	Map<VectorXcd> map_bp(bp.vector, bp.vlen);

	EXPECT_NEAR((map_b.cast<complex128_t>()-map_bp).norm()/map_b.norm(),
		0.0, 1E-15);


}

TEST(DirectLinearSolverComplex, solve_QR_COLPERM)
{
	const index_t size=2;
	SGMatrix<complex128_t> m(size, size);
	m(0,0)=complex128_t(2.0);
	m(0,1)=complex128_t(1.0, 2.0);
	m(1,0)=complex128_t(1.0, 2.0);
	m(1,1)=complex128_t(3.0);

	auto A
		=std::make_shared<DenseMatrixOperator<complex128_t>>(m);

	SGVector<float64_t> b(size);
	b.set_const(1.0);

	DirectLinearSolverComplex solver(DS_QR_COLPERM);
	SGVector<complex128_t> x
		=solver.solve(A->as<LinearOperator<complex128_t>>(), b);

	SGVector<complex128_t> bp=A->apply(x);
	Map<VectorXd> map_b(b.vector, b.vlen);
	Map<VectorXcd> map_bp(bp.vector, bp.vlen);

	EXPECT_NEAR((map_b.cast<complex128_t>()-map_bp).norm()/map_b.norm(),
		0.0, 1E-15);


}

TEST(DirectLinearSolverComplex, solve_QR_FULLPERM)
{
	const index_t size=2;
	SGMatrix<complex128_t> m(size, size);
	m(0,0)=complex128_t(2.0);
	m(0,1)=complex128_t(1.0, 2.0);
	m(1,0)=complex128_t(1.0, 2.0);
	m(1,1)=complex128_t(3.0);

	auto A
		=std::make_shared<DenseMatrixOperator<complex128_t>>(m);

	SGVector<float64_t> b(size);
	b.set_const(1.0);

	DirectLinearSolverComplex solver(DS_QR_FULLPERM);
	SGVector<complex128_t> x
		=solver.solve(A->as<LinearOperator<complex128_t>>(), b);

	SGVector<complex128_t> bp=A->apply(x);
	Map<VectorXd> map_b(b.vector, b.vlen);
	Map<VectorXcd> map_bp(bp.vector, bp.vlen);

	EXPECT_NEAR((map_b.cast<complex128_t>()-map_bp).norm()/map_b.norm(),
		0.0, 1E-15);


}

TEST(DirectLinearSolverComplex, solve_LLT)
{
	const index_t size=2;
	SGMatrix<complex128_t> m(size, size);
	// LLT doesn't work on non-symmetric matrices
	m(0,0)=complex128_t(2.0, 0.0);
	m(0,1)=complex128_t(1.0, 0.0);
	m(1,0)=complex128_t(1.0, 0.0);
	m(1,1)=complex128_t(2.5, 0.0);

	auto A
		=std::make_shared<DenseMatrixOperator<complex128_t>>(m);

	SGVector<float64_t> b(size);
	b.set_const(1.0);

	DirectLinearSolverComplex solver(DS_LLT);
	SGVector<complex128_t> x
		=solver.solve(A->as<LinearOperator<complex128_t>>(), b);

	SGVector<complex128_t> bp=A->apply(x);
	Map<VectorXd> map_b(b.vector, b.vlen);
	Map<VectorXcd> map_bp(bp.vector, bp.vlen);

	EXPECT_NEAR((map_b.cast<complex128_t>()-map_bp).norm()/map_b.norm(),
		0.0, 1E-15);


}
