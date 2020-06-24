/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soumyajit De, Viktor Gal, Thoralf Klein, Heiko Strathmann,
 *          Bjoern Esser, Shubham Shukla, Pan Deng
 */
#include <gtest/gtest.h>

#include <shogun/lib/common.h>

#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGSparseMatrix.h>
#include <shogun/features/SparseFeatures.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/linalg/linop/DenseMatrixOperator.h>
#include <shogun/mathematics/linalg/linop/SparseMatrixOperator.h>
#include <shogun/mathematics/linalg/linsolver/DirectLinearSolverComplex.h>
#include <shogun/mathematics/linalg/linsolver/CGMShiftedFamilySolver.h>
#include <shogun/mathematics/linalg/linsolver/ConjugateOrthogonalCGSolver.h>
#include <shogun/mathematics/linalg/eigsolver/DirectEigenSolver.h>
#include <shogun/mathematics/linalg/ratapprox/tracesampler/NormalSampler.h>
#include <shogun/mathematics/linalg/ratapprox/logdet/opfunc/LogRationalApproximationIndividual.h>
#include <shogun/mathematics/linalg/ratapprox/logdet/opfunc/LogRationalApproximationCGM.h>
#include <unsupported/Eigen/MatrixFunctions>

using namespace shogun;
using namespace Eigen;

#ifdef USE_GPL_SHOGUN
TEST(RationalApproximation, precompute)
{
	const index_t size=2;
	SGMatrix<float64_t> m(size, size);
	m(0,0)=2.0;
	m(0,1)=1.0;
	m(1,0)=1.0;
	m(1,1)=3.0;

	auto op=std::make_shared<DenseMatrixOperator<float64_t>>(m);


	auto eig_solver=std::make_shared<DirectEigenSolver>(op);


	auto linear_solver=std::make_shared<DirectLinearSolverComplex>();


	auto op_func =
	    std::make_shared<LogRationalApproximationIndividual>(
	        op, eig_solver,
	        linear_solver->as<LinearSolver<complex128_t, float64_t>>(), 0);

	op_func->set_num_shifts(5);

	op_func->precompute();

	SGVector<complex128_t> shifts=op_func->get_shifts();
	SGVector<complex128_t> weights=op_func->get_weights();
	float64_t const_multiplier=op_func->get_constant_multiplier();

	Map<VectorXcd> map_shifts(shifts.vector, shifts.vlen);
	Map<VectorXcd> map_weights(weights.vector, weights.vlen);

	// reference values are generated using KRYLSTAT
	SGVector<complex128_t> ref_shifts(5);
	ref_shifts[0]=complex128_t(0.51827127849765364243, 0.23609847245566201179);
	ref_shifts[1]=complex128_t(0.44961096840887382342, 0.86844451701724056925);
	ref_shifts[2]=complex128_t(0.52786404500042061194, 2.17286896751640146164);
	ref_shifts[3]=complex128_t(2.35067127618097071462, 4.54043100490560203042);
	ref_shifts[4]=complex128_t(7.98944200008961935566, 3.63959017266438733529);
	Map<VectorXcd> map_ref_shifts(ref_shifts.vector, ref_shifts.vlen);
	SGVector<complex128_t> ref_weights(5);
	ref_weights[0]=complex128_t(-0.01647563566875611188, -0.01058494296357846871);
	ref_weights[1]=complex128_t(-0.01690640878366324318, 0.02513114861539664235);
	ref_weights[2]=complex128_t(0.02229379592072704488, 0.03691476590076240433);
	ref_weights[3]=complex128_t(0.05440205028428418688, 0.00100377381669515997);
	ref_weights[4]=complex128_t(0.03177788684575824640, -0.05246446606420653719);
	Map<VectorXcd> map_ref_weights(ref_weights.vector, ref_weights.vlen);

#ifdef HAVE_ARPREC
	EXPECT_NEAR(const_multiplier, -10.02791094628079981987, 1E-19);
	EXPECT_NEAR(map_shifts.norm(), map_ref_shifts.norm(), 1E-14);
	EXPECT_NEAR(map_weights.norm(), map_ref_weights.norm(), 1E-14);
#else
	EXPECT_NEAR(const_multiplier, -10.02791094628079981987, 1E-15);
	EXPECT_NEAR(map_shifts.norm(), map_ref_shifts.norm(), 1E-12);
	EXPECT_NEAR(map_weights.norm(), map_ref_weights.norm(), 1E-12);
#endif





}

TEST(RationalApproximation, trace_accuracy)
{
	const index_t size=5;
	SGMatrix<float64_t> m(size, size);
	m.set_const(0.0);
	float64_t coeff=0.001;
	for (index_t i=0; i<size; ++i)
	{
		m(i,i)=coeff;
		coeff*=10;
	}

	// create the operator
	auto op=std::make_shared<DenseMatrixOperator<float64_t>>(m);


	// create the eigen solver for finding max/min eigenvalues
	auto eig_solver=std::make_shared<DirectEigenSolver>(op);


	// create the direct linear solver for solving the systems that generates from
	// rational approximation of the operator function
	auto linear_solver=std::make_shared<DirectLinearSolverComplex>();


	// compute the number of shifts to assure a given accuracy
	float64_t accuracy=1E-19;

	// create the operator function that extracts the trace
	// of the approximation of log of the linear operator
	auto op_func =
	    std::make_shared<LogRationalApproximationIndividual>(
	        op, eig_solver, linear_solver, accuracy);


	op_func->precompute();
	float64_t result = 0.0;
	// extract the trace of approximation of log using basis vectors
	for (index_t i=0; i<size; ++i)
	{
		SGVector<float64_t> s(size);
		s.set_const(0.0);
		s[i]=1.0;
		result += op_func->compute(s);
	}

	// compute the trace of log(m) using Eigen3 that uses Schur-Parlett algorithm
	Map<MatrixXd> eig_m(m.matrix, m.num_rows, m.num_cols);
	float64_t trace_log_m=eig_m.log().diagonal().sum();

#ifdef HAVE_ARPREC
	EXPECT_NEAR(result, trace_log_m, 1E-13);
#else
	EXPECT_NEAR(result, trace_log_m, 1E-07);
#endif





}

TEST(RationalApproximation, compare_direct_vs_cocg_accuracy)
{
	const index_t size=2;
	SGMatrix<float64_t> m(size, size);
	m(0,0)=1.0;
	m(0,1)=0.0;
	m(1,0)=0.0;
	m(1,1)=100000.0;

	auto op=std::make_shared<DenseMatrixOperator<float64_t>>(m);


	auto eig_solver=std::make_shared<DirectEigenSolver>(op);


	auto dense_solver=std::make_shared<DirectLinearSolverComplex>();


	auto sparse_solver
		=std::make_shared<ConjugateOrthogonalCGSolver>();

	auto op_func =
	    std::make_shared<LogRationalApproximationIndividual>(
	        op, eig_solver,
	        dense_solver->as<LinearSolver<complex128_t, float64_t>>(), 0);

	op_func->set_num_shifts(4);

	op_func->precompute();

	SGVector<complex128_t> shifts=op_func->get_shifts();

	auto trace_sampler=std::make_shared<NormalSampler>(size);
	trace_sampler->precompute();

	// create complex copies of operators, complex_dense/sparse
	auto complex_dense
		=std::shared_ptr<DenseMatrixOperator<complex128_t>>(static_cast<DenseMatrixOperator<complex128_t>*>(*op));

	for (index_t i=0; i<shifts.vlen; ++i)
	{
		SGVector<float64_t> sample=trace_sampler->sample(0);

		auto shifted_dense
			=std::make_shared<DenseMatrixOperator<complex128_t>>(*complex_dense);

		SGVector<complex128_t> diag=shifted_dense->get_diagonal();
		for (index_t j=0; j<diag.vlen; ++j)
			diag[j]-=shifts[i];

		shifted_dense->set_diagonal(diag);

		SGMatrix<complex128_t> shifted_m=shifted_dense->get_matrix_operator();

		SparseFeatures<complex128_t> feat(shifted_m);
		SGSparseMatrix<complex128_t> shifted_sm=feat.get_sparse_feature_matrix();
		auto shifted_sparse
			=std::make_shared<SparseMatrixOperator<complex128_t>>(shifted_sm);

		SGVector<complex128_t> xd=dense_solver->solve(shifted_dense, sample);
		SGVector<complex128_t> xs=sparse_solver->solve(shifted_sparse, sample);

		Map<VectorXcd> map_xd(xd.vector, xd.vlen);
		Map<VectorXcd> map_xs(xs.vector, xs.vlen);

		EXPECT_NEAR((map_xd-map_xs).norm(), 0.0, 1E-10);



	}








}

TEST(RationalApproximation, trace_accuracy_cg_m)
{
	const index_t size=5;
	SGMatrix<float64_t> m(size, size);
	m.set_const(0.0);

	float64_t coeff=0.001;
	for (index_t i=0; i<size; ++i)
	{
		m(i,i)=coeff;
		coeff*=10;
	}

	// create the operator
	auto op=std::make_shared<DenseMatrixOperator<float64_t>>(m);


	// create the eigen solver for finding max/min eigenvalues
	auto eig_solver=std::make_shared<DirectEigenSolver>(op);


	// create the direct linear solver for solving the systems that generates from
	// rational approximation of the operator function
	auto linear_solver=std::make_shared<CGMShiftedFamilySolver>();


	// compute the number of shifts to assure a given accuracy
	float64_t accuracy=1E-19;

	// create the operator function that extracts the trace
	// of the approximation of log of the linear operator
	auto op_func = std::make_shared<LogRationalApproximationCGM>(
	    op, eig_solver, linear_solver, accuracy);


	op_func->precompute();
	float64_t result = 0.0;
	// extract the trace of approximation of log using basis vectors
	for (index_t i=0; i<size; ++i)
	{
		SGVector<float64_t> s(size);
		s.set_const(0.0);
		s[i]=1.0;
		result += op_func->compute(s);
	}

	// compute the trace of log(m) using Eigen3 that uses Schur-Parlett algorithm
	Map<MatrixXd> eig_m(m.matrix, m.num_rows, m.num_cols);
	float64_t trace_log_m=eig_m.log().diagonal().sum();

#ifdef HAVE_ARPREC
	EXPECT_NEAR(result, trace_log_m, 1E-13);
#else
	EXPECT_NEAR(result, trace_log_m, 1E-07);
#endif
}
#endif //USE_GPL_SHOGUN
