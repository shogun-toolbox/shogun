/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soumyajit De, Sunil Mahendrakar, Pan Deng, Bjoern Esser, Viktor Gal
 */
#include <gtest/gtest.h>

#include <shogun/lib/common.h>

#ifdef HAVE_LAPACK

#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGSparseMatrix.h>
#include <shogun/features/SparseFeatures.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/linalg/linop/SparseMatrixOperator.h>
#include <shogun/mathematics/linalg/linop/DenseMatrixOperator.h>
#include <shogun/mathematics/linalg/eigsolver/DirectEigenSolver.h>
#include <shogun/mathematics/linalg/eigsolver/LanczosEigenSolver.h>
#include <shogun/mathematics/NormalDistribution.h>
#include <shogun/mathematics/UniformRealDistribution.h>

#include <random>

using namespace shogun;

TEST(LanczosEigenSolver, compute)
{
	const int32_t seed = 10;
	const int32_t size=4;
	SGMatrix<float64_t> m(size, size);
	std::mt19937_64 prng(seed);

	UniformRealDistribution<float64_t> uniform_real_dist;
	m.set_const(uniform_real_dist(prng, {50.0, 100.0}));

	// Hermintian matrix
	for (index_t i=0; i<size; ++i)
		m(i,i)=uniform_real_dist(prng, {100.0, 10000.0});

	// Creating sparse linear operator to use with Lanczos
	SparseFeatures<float64_t> feat(m);
	SGSparseMatrix<float64_t> mat=feat.get_sparse_feature_matrix();
	auto A=std::make_shared<SparseMatrixOperator<float64_t>>(mat);
	std::shared_ptr<EigenSolver> eig_solver=std::make_shared<LanczosEigenSolver>(A);
	eig_solver->compute();

	float64_t lanc_max_eig=eig_solver->get_max_eigenvalue();
	float64_t lanc_min_eig=eig_solver->get_min_eigenvalue();

	// create dense linear operator to use with direct eigensolver
	auto B=std::make_shared<DenseMatrixOperator<float64_t>>(m);

	eig_solver=std::make_shared<DirectEigenSolver>(B);
	eig_solver->compute();

	float64_t dir_max_eig=eig_solver->get_max_eigenvalue();
	float64_t dir_min_eig=eig_solver->get_min_eigenvalue();

	// compare these two
	EXPECT_NEAR(Math::abs(lanc_max_eig-dir_max_eig), 0.0, 1E-6);
	EXPECT_NEAR(Math::abs(lanc_min_eig-dir_min_eig), 0.0, 1E-6);
}

TEST(LanczosEigenSolver, compute_big_diag_matrix)
{
	int32_t seed = 10;
	float64_t difficulty=4;
	float64_t min_eigenvalue=0.0001;

	// create a sparse matrix
	const index_t size=100;
	SGSparseMatrix<float64_t> sm(size, size);
	auto op=std::make_shared<SparseMatrixOperator<float64_t>>(sm);

	// set its diagonal
	SGVector<float64_t> diag(size);
	std::mt19937_64 prng(seed);
	NormalDistribution<float64_t> normal_dist;
	for (index_t i=0; i<size; ++i)
	{
		diag[i]=Math::pow(Math::abs(normal_dist(prng)), difficulty)
			+min_eigenvalue;
	}
	op->set_diagonal(diag);

	auto eig_solver=std::make_shared<LanczosEigenSolver>(op);

	eig_solver->compute();

	// test eigenvalues
	Eigen::Map<Eigen::VectorXd> diag_map(diag.vector, diag.vlen);
	float64_t actual_min_eig=diag_map.minCoeff();
	float64_t actual_max_eig=diag_map.maxCoeff();
	float64_t computed_min_eig=eig_solver->get_min_eigenvalue();
	float64_t computed_max_eig=eig_solver->get_max_eigenvalue();
	EXPECT_NEAR(actual_min_eig, computed_min_eig, 1E-4);
	EXPECT_NEAR(actual_max_eig, computed_max_eig, 1E-4);



}

TEST(LanczosEigenSolver, set_eigenvalues_externally)
{
	const index_t size=2;
	SGMatrix<float64_t> m(size, size);
	m(0,0)=1;
	m(1,1)=2;
	auto A=std::make_shared<DenseMatrixOperator<float64_t>>(m);

	float64_t min_eigenvalue=0.0001;
	float64_t max_eigenvalue=100000.0;
	auto eig_solver=std::make_shared<LanczosEigenSolver>(A);
	eig_solver->set_min_eigenvalue(min_eigenvalue);
	eig_solver->set_max_eigenvalue(max_eigenvalue);

	eig_solver->compute();
	EXPECT_NEAR(eig_solver->get_min_eigenvalue(), min_eigenvalue, 1E-16);
	EXPECT_NEAR(eig_solver->get_max_eigenvalue(), max_eigenvalue, 1E-16);



}
#endif // HAVE_LAPACK

