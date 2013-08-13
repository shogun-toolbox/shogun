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
#include <shogun/mathematics/Statistics.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/features/SparseFeatures.h>
#include <shogun/mathematics/logdet/DenseMatrixOperator.h>
#include <shogun/mathematics/logdet/SparseMatrixOperator.h>
#include <shogun/mathematics/logdet/DenseMatrixExactLog.h>
#include <shogun/mathematics/logdet/LogRationalApproximationIndividual.h>
#include <shogun/mathematics/logdet/NormalSampler.h>
#include <shogun/mathematics/logdet/DirectEigenSolver.h>
#include <shogun/mathematics/logdet/LanczosEigenSolver.h>
#include <shogun/mathematics/logdet/DirectLinearSolverComplex.h>
#include <shogun/mathematics/logdet/ConjugateOrthogonalCGSolver.h>
#include <shogun/mathematics/logdet/LogDetEstimator.h>
#include <shogun/lib/computation/job/ScalarResult.h>
#include <shogun/lib/computation/engine/SerialComputationEngine.h>
#include <gtest/gtest.h>

using namespace shogun;

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

	CDenseMatrixOperator<float64_t>* op=new CDenseMatrixOperator<float64_t>(mat);
	SG_REF(op);

	CDirectEigenSolver* eig_solver=new CDirectEigenSolver(op);
	SG_REF(eig_solver);

	CDirectLinearSolverComplex* linear_solver=new CDirectLinearSolverComplex();
	SG_REF(linear_solver);

	CLogRationalApproximationIndividual *op_func
		=new CLogRationalApproximationIndividual(
			op, e, eig_solver,
			(CLinearSolver<complex64_t, float64_t>*)linear_solver, 8);
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
#endif // HAVE_EIGEN3

