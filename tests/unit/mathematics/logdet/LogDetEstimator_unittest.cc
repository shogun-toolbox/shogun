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
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/Random.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGSparseMatrix.h>
#include <shogun/features/SparseFeatures.h>
#include <shogun/mathematics/logdet/DenseMatrixOperator.h>
#include <shogun/mathematics/logdet/SparseMatrixOperator.h>
#include <shogun/mathematics/logdet/DenseMatrixExactLog.h>
#include <shogun/mathematics/logdet/LogRationalApproximationIndividual.h>
#include <shogun/mathematics/logdet/LogRationalApproximationCGM.h>
#include <shogun/mathematics/logdet/NormalSampler.h>
#include <shogun/mathematics/logdet/ProbingSampler.h>
#include <shogun/mathematics/logdet/DirectEigenSolver.h>
#include <shogun/mathematics/logdet/LanczosEigenSolver.h>
#include <shogun/mathematics/logdet/DirectLinearSolverComplex.h>
#include <shogun/mathematics/logdet/CGMShiftedFamilySolver.h>
#include <shogun/mathematics/logdet/LogDetEstimator.h>
#include <shogun/lib/computation/job/ScalarResult.h>
#include <shogun/lib/computation/engine/SerialComputationEngine.h>
#include <gtest/gtest.h>

using namespace shogun;
using namespace Eigen;

#if EIGEN_VERSION_AT_LEAST(3,1,0)
TEST(LogDetEstimator, sample)
{
	CSerialComputationEngine* e=new CSerialComputationEngine;
	SG_REF(e);
	
	const index_t size=2;
	SGMatrix<float64_t> mat(size, size);
	mat(0,0)=2.0;
	mat(0,1)=1.0;
	mat(1,0)=1.0;
	mat(1,1)=3.0;

	CDenseMatrixOperator<float64_t>* op=new CDenseMatrixOperator<float64_t>(mat);
	SG_REF(op);

	CDenseMatrixExactLog *op_func=new CDenseMatrixExactLog(op, e);
	SG_REF(op_func);

	CNormalSampler* trace_sampler=new CNormalSampler(size);
	SG_REF(trace_sampler);

	CLogDetEstimator estimator(trace_sampler, op_func, e);
	const index_t num_estimates=5000;
	SGVector<float64_t> estimates=estimator.sample(num_estimates);
	
	float64_t result=0.0;
	for (index_t i=0; i<num_estimates; ++i)
		result+=estimates[i];
	result/=num_estimates;

	EXPECT_NEAR(result, 1.60943791243410050384, 0.1);

	SG_UNREF(trace_sampler);
	SG_UNREF(op_func);
	SG_UNREF(op);
	SG_UNREF(e);
}
#endif // EIGEN_VERSION_AT_LEAST(3,1,0)

TEST(LogDetEstimator, sample_ratapp_dense)
{
	CSerialComputationEngine* e=new CSerialComputationEngine;
	SG_REF(e);
	
	const index_t size=2;
	SGMatrix<float64_t> mat(size, size);
	mat(0,0)=1.0;
	mat(0,1)=0.5;
	mat(1,0)=0.5;
	mat(1,1)=1000.0;

	float64_t accuracy=1E-15;
	CDenseMatrixOperator<float64_t>* op=new CDenseMatrixOperator<float64_t>(mat);
	SG_REF(op);

	CDirectEigenSolver* eig_solver=new CDirectEigenSolver(op);
	SG_REF(eig_solver);

	CDirectLinearSolverComplex* linear_solver=new CDirectLinearSolverComplex();
	SG_REF(linear_solver);

	CLogRationalApproximationIndividual *op_func
		=new CLogRationalApproximationIndividual(
			op, e, eig_solver,
			(CLinearSolver<complex64_t, float64_t>*)linear_solver, accuracy);
	SG_REF(op_func);
	
	CNormalSampler* trace_sampler=new CNormalSampler(size);
	SG_REF(trace_sampler);

	CLogDetEstimator estimator(trace_sampler, op_func, e);
	const index_t num_estimates=500;
	SGVector<float64_t> estimates=estimator.sample(num_estimates);
	
	float64_t result=0.0;
	for (index_t i=0; i<num_estimates; ++i)
		result+=estimates[i];
	result/=num_estimates;

	EXPECT_NEAR(result, CStatistics::log_det(mat), 2.0);

	SG_UNREF(trace_sampler);
	SG_UNREF(eig_solver);
	SG_UNREF(linear_solver);
	SG_UNREF(op_func);
	SG_UNREF(op);
	SG_UNREF(e);
}

#ifdef HAVE_COLPACK
#ifdef HAVE_LAPACK
TEST(LogDetEstimator, sample_ratapp_probing_sampler)
{
	CSerialComputationEngine* e=new CSerialComputationEngine;
	SG_REF(e);
	
	const index_t size=16;
	SGMatrix<float64_t> mat(size, size);
	mat.set_const(0.0);
	for (index_t i=0; i<size; ++i)
	{
		float64_t value=CMath::abs(sg_rand->std_normal_distrib())*1000;
		mat(i,i)=value<1.0?10.0:value;
	}

	mat(0,5)=mat(5,0)=1.0;
	mat(0,7)=mat(7,0)=1.0;
	mat(0,11)=mat(11,0)=1.0;
	mat(1,8)=mat(8,1)=1.0;
	mat(1,10)=mat(10,1)=1.0;
	mat(1,11)=mat(11,1)=1.0;
	mat(1,12)=mat(12,1)=1.0;
	mat(2,8)=mat(8,2)=1.0;
	mat(2,11)=mat(11,2)=1.0;
	mat(2,13)=mat(13,2)=1.0;
	mat(2,14)=mat(14,2)=1.0;
	mat(3,8)=mat(8,3)=1.0;
	mat(3,12)=mat(12,3)=1.0;
	mat(3,15)=mat(15,3)=1.0;
	mat(4,8)=mat(8,4)=1.0;
	mat(4,14)=mat(14,4)=1.0;
	mat(4,15)=mat(15,4)=1.0;
	mat(5,11)=mat(11,5)=1.0;
	mat(5,10)=mat(10,5)=1.0;
	mat(6,10)=mat(10,6)=1.0;
	mat(6,12)=mat(12,6)=1.0;
	mat(7,11)=mat(11,7)=1.0;
	mat(7,13)=mat(13,7)=1.0;
	mat(8,11)=mat(11,8)=1.0;
	mat(8,15)=mat(15,8)=1.0;
	mat(9,13)=mat(13,9)=1.0;
	mat(9,14)=mat(14,9)=1.0;

	float64_t actual_result=CStatistics::log_det(mat);
	float64_t accuracy=1E-15;

	CSparseFeatures<float64_t> feat(mat);
	SGSparseMatrix<float64_t> sm=feat.get_sparse_feature_matrix();

	CSparseMatrixOperator<float64_t>* op=new CSparseMatrixOperator<float64_t>(sm);
	SG_REF(op);
	CDenseMatrixOperator<float64_t>* opd=new CDenseMatrixOperator<float64_t>(mat);
	SG_REF(opd);

	CLanczosEigenSolver* eig_solver=new CLanczosEigenSolver(op);
	SG_REF(eig_solver);

	CDirectLinearSolverComplex* linear_solver=new CDirectLinearSolverComplex();
	SG_REF(linear_solver);

	CLogRationalApproximationIndividual *op_func
		=new CLogRationalApproximationIndividual
		(opd, e, eig_solver, (CLinearSolver<complex64_t, float64_t>*)linear_solver, accuracy);
	SG_REF(op_func);

	CProbingSampler* trace_sampler=new CProbingSampler(op, 1, NATURAL, DISTANCE_TWO);
	SG_REF(trace_sampler);

	CLogDetEstimator estimator(trace_sampler, op_func, e);
	const index_t num_estimates=10;
	SGVector<float64_t> estimates=estimator.sample(num_estimates);

	float64_t result=0.0;
	for (index_t i=0; i<num_estimates; ++i)
		result+=estimates[i];
	result/=num_estimates;

	EXPECT_NEAR(result, actual_result, 1E-3);

	SG_UNREF(trace_sampler);
	SG_UNREF(eig_solver);
	SG_UNREF(linear_solver);
	SG_UNREF(op_func);
	SG_UNREF(op);
	SG_UNREF(opd);
	SG_UNREF(e);
}

TEST(LogDetEstimator, sample_ratapp_probing_sampler_cgm)
{
	CSerialComputationEngine* e=new CSerialComputationEngine;
	SG_REF(e);
	
	const index_t size=16;
	SGMatrix<float64_t> mat(size, size);
	mat.set_const(0.0);
	for (index_t i=0; i<size; ++i)
	{
		float64_t value=CMath::abs(sg_rand->std_normal_distrib())*1000;
		mat(i,i)=value<1.0?10.0:value;
	}

	mat(0,5)=mat(5,0)=1.0;
	mat(0,7)=mat(7,0)=1.0;
	mat(0,11)=mat(11,0)=1.0;
	mat(1,8)=mat(8,1)=1.0;
	mat(1,10)=mat(10,1)=1.0;
	mat(1,11)=mat(11,1)=1.0;
	mat(1,12)=mat(12,1)=1.0;
	mat(2,8)=mat(8,2)=1.0;
	mat(2,11)=mat(11,2)=1.0;
	mat(2,13)=mat(13,2)=1.0;
	mat(2,14)=mat(14,2)=1.0;
	mat(3,8)=mat(8,3)=1.0;
	mat(3,12)=mat(12,3)=1.0;
	mat(3,15)=mat(15,3)=1.0;
	mat(4,8)=mat(8,4)=1.0;
	mat(4,14)=mat(14,4)=1.0;
	mat(4,15)=mat(15,4)=1.0;
	mat(5,11)=mat(11,5)=1.0;
	mat(5,10)=mat(10,5)=1.0;
	mat(6,10)=mat(10,6)=1.0;
	mat(6,12)=mat(12,6)=1.0;
	mat(7,11)=mat(11,7)=1.0;
	mat(7,13)=mat(13,7)=1.0;
	mat(8,11)=mat(11,8)=1.0;
	mat(8,15)=mat(15,8)=1.0;
	mat(9,13)=mat(13,9)=1.0;
	mat(9,14)=mat(14,9)=1.0;

	float64_t actual_result=CStatistics::log_det(mat);
	float64_t accuracy=1E-15;

	CSparseFeatures<float64_t> feat(mat);
	SGSparseMatrix<float64_t> sm=feat.get_sparse_feature_matrix();

	CSparseMatrixOperator<float64_t>* op=new CSparseMatrixOperator<float64_t>(sm);
	SG_REF(op);

	CLanczosEigenSolver* eig_solver=new CLanczosEigenSolver(op);
	SG_REF(eig_solver);

	CCGMShiftedFamilySolver* linear_solver=new CCGMShiftedFamilySolver();
	SG_REF(linear_solver);

	CLogRationalApproximationCGM *op_func
		=new CLogRationalApproximationCGM(op, e, eig_solver, linear_solver, accuracy);
	SG_REF(op_func);

	CProbingSampler* trace_sampler=new CProbingSampler(op, 1, NATURAL, DISTANCE_TWO);
	SG_REF(trace_sampler);

	CLogDetEstimator estimator(trace_sampler, op_func, e);
	const index_t num_estimates=10;
	SGVector<float64_t> estimates=estimator.sample(num_estimates);

	float64_t result=0.0;
	for (index_t i=0; i<num_estimates; ++i)
		result+=estimates[i];
	result/=num_estimates;

	EXPECT_NEAR(result, actual_result, 1E-3);

	SG_UNREF(trace_sampler);
	SG_UNREF(eig_solver);
	SG_UNREF(linear_solver);
	SG_UNREF(op_func);
	SG_UNREF(op);
	SG_UNREF(e);
}

TEST(LogDetEstimator, sample_ratapp_big_diag_matrix)
{
	CSerialComputationEngine* e=new CSerialComputationEngine;
	SG_REF(e);

	float64_t difficulty=2;
	float64_t accuracy=1E-15;
	float64_t min_eigenvalue=0.001;

	// create a sparse matrix	
	const index_t size=10000;
	SGSparseMatrix<float64_t> sm(size, size);
	CSparseMatrixOperator<float64_t>* op=new CSparseMatrixOperator<float64_t>(sm);
	SG_REF(op);

	// set its diagonal
	SGVector<float64_t> diag(size);
	for (index_t i=0; i<size; ++i)
	{
		diag[i]=CMath::pow(CMath::abs(sg_rand->std_normal_distrib()), difficulty)
			+min_eigenvalue;
	}
	op->set_diagonal(diag);

	CLanczosEigenSolver* eig_solver=new CLanczosEigenSolver(op);
	SG_REF(eig_solver);

	CCGMShiftedFamilySolver* linear_solver=new CCGMShiftedFamilySolver();
	SG_REF(linear_solver);

	CLogRationalApproximationCGM *op_func
		=new CLogRationalApproximationCGM(op, e, eig_solver, linear_solver, accuracy);
	SG_REF(op_func);

	CProbingSampler* trace_sampler=new CProbingSampler(op);
	SG_REF(trace_sampler);

	CLogDetEstimator estimator(trace_sampler, op_func, e);
	const index_t num_estimates=1;
	SGVector<float64_t> estimates=estimator.sample(num_estimates);

	float64_t result=0.0;
	for (index_t i=0; i<num_estimates; ++i)
		result+=estimates[i];
	result/=num_estimates;

	// test the log-det samples
	sm=op->get_matrix_operator();
	float64_t actual_result=CStatistics::log_det(sm);
	EXPECT_NEAR(result, actual_result, 1E-2);

	SG_UNREF(trace_sampler);
	SG_UNREF(eig_solver);
	SG_UNREF(linear_solver);
	SG_UNREF(op_func);
	SG_UNREF(op);
	SG_UNREF(e);
}

TEST(LogDetEstimator, sample_ratapp_big_matrix)
{
	CSerialComputationEngine* e=new CSerialComputationEngine;
	SG_REF(e);

	float64_t difficulty=2;
	float64_t accuracy=1E-15;
	float64_t min_eigenvalue=0.001;

	// create a sparse matrix	
	const index_t size=10000;
	SGSparseMatrix<float64_t> sm(size, size);

	// set its diagonal
	SGVector<float64_t> diag(size);
	for (index_t i=0; i<size; ++i)
	{
		sm(i,i)=CMath::pow(CMath::abs(sg_rand->std_normal_distrib()), difficulty)
			+min_eigenvalue;
	}
	// set its subdiagonal
	float64_t entry=min_eigenvalue/2;
	for (index_t i=0; i<size-1; ++i)
	{
		sm(i,i+1)=entry;
		sm(i+1,i)=entry;
	}

	CSparseMatrixOperator<float64_t>* op=new CSparseMatrixOperator<float64_t>(sm);
	SG_REF(op);

	CLanczosEigenSolver* eig_solver=new CLanczosEigenSolver(op);
	SG_REF(eig_solver);

	CCGMShiftedFamilySolver* linear_solver=new CCGMShiftedFamilySolver();
	linear_solver->set_iteration_limit(2000);
	SG_REF(linear_solver);

	CLogRationalApproximationCGM *op_func
		=new CLogRationalApproximationCGM(op, e, eig_solver, linear_solver, accuracy);
	SG_REF(op_func);

	CProbingSampler* trace_sampler=new CProbingSampler(op);
	SG_REF(trace_sampler);

	CLogDetEstimator estimator(trace_sampler, op_func, e);
	const index_t num_estimates=1;
	SGVector<float64_t> estimates=estimator.sample(num_estimates);

	float64_t result=0.0;
	for (index_t i=0; i<num_estimates; ++i)
		result+=estimates[i];
	result/=num_estimates;

	// test the log-det samples
	sm=op->get_matrix_operator();
	float64_t actual_result=CStatistics::log_det(sm);
	EXPECT_NEAR(result, actual_result, 0.01);

	SG_UNREF(trace_sampler);
	SG_UNREF(eig_solver);
	SG_UNREF(linear_solver);
	SG_UNREF(op_func);
	SG_UNREF(op);
	SG_UNREF(e);
}
#endif // HAVE_LAPACK
#endif // HAVE_COLPACK
#endif // HAVE_EIGEN3
