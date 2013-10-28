/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Soumyajit De
 */

#include <shogun/lib/config.h>
#ifdef HAVE_EIGEN3
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/linalg/linop/DenseMatrixOperator.h>
#include <shogun/mathematics/linalg/linsolver/DirectLinearSolverComplex.h>
#include <shogun/mathematics/linalg/linsolver/ConjugateOrthogonalCGSolver.h>
#include <shogun/lib/computation/jobresult/ScalarResult.h>
#include <shogun/mathematics/linalg/ratapprox/logdet/computation/aggregator/IndividualJobResultAggregator.h>
#include <shogun/mathematics/linalg/ratapprox/logdet/computation/job/RationalApproximationIndividualJob.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(RationalApproximationIndividualJob, compute_direct)
{
	const index_t size=2;
	SGMatrix<complex128_t> m(size, size);
	m(0,0)=complex128_t(2.0, 1.0);
	m(0,1)=complex128_t(1.0);
	m(1,0)=complex128_t(1.0);
	m(1,1)=complex128_t(3.0, 1.0);

	SGMatrix<float64_t> mi(size, size);
	mi(0,0)=1.0;
	mi(0,1)=0.0;
	mi(1,0)=0.0;
	mi(1,1)=1.0;
	CDenseMatrixOperator<float64_t>* identity=new CDenseMatrixOperator<float64_t>(mi);
	SG_REF(identity);

	SG_SDEBUG("creating op\n");
	CDenseMatrixOperator<complex128_t>* op=new CDenseMatrixOperator<complex128_t>(m);
	SG_REF(op);

	const float64_t const_multiplier=2.0;
	const complex128_t weight=complex128_t(1.0, 1.0);
	SGVector<float64_t> s(size);
	s.set_const(1.0);
	// creating aggregator
	CIndividualJobResultAggregator* agg=new CIndividualJobResultAggregator(
		identity, s, const_multiplier);
	SG_REF(agg);

	// creating complex linear solver
	CDirectLinearSolverComplex* solver=new CDirectLinearSolverComplex();
	SG_REF(solver);

	CRationalApproximationIndividualJob* job
		=new CRationalApproximationIndividualJob(agg,
			(CLinearSolver<complex128_t, float64_t>*)solver, op, s, weight);
	SG_REF(job);
	job->compute();
	SG_UNREF(job);
	agg->finalize();
	CScalarResult<float64_t>* r=dynamic_cast<CScalarResult<float64_t>*>
		(agg->get_final_result());
	float64_t result=r->get_result();
	SG_UNREF(agg);
	SG_UNREF(op);
	SG_UNREF(identity);
	SG_UNREF(solver);

	EXPECT_NEAR(result, -0.73170731707317049, 1E-15);
}

TEST(RationalApproximationIndividualJob, compute_cocg)
{
	const index_t size=2;
	SGMatrix<complex128_t> m(size, size);
	m(0,0)=complex128_t(2.0, 1.0);
	m(0,1)=complex128_t(1.0);
	m(1,0)=complex128_t(1.0);
	m(1,1)=complex128_t(3.0, 1.0);

	SGMatrix<float64_t> mi(size, size);
	mi(0,0)=1.0;
	mi(0,1)=0.0;
	mi(1,0)=0.0;
	mi(1,1)=1.0;
	CDenseMatrixOperator<float64_t>* identity=new CDenseMatrixOperator<float64_t>(mi);
	SG_REF(identity);

	SG_SDEBUG("creating op\n");
	CDenseMatrixOperator<complex128_t>* op=new CDenseMatrixOperator<complex128_t>(m);
	SG_REF(op);

	const float64_t const_multiplier=2.0;
	const complex128_t weight=complex128_t(1.0, 1.0);
	SGVector<float64_t> s(size);
	s.set_const(1.0);
	// creating aggregator
	CIndividualJobResultAggregator* agg=new CIndividualJobResultAggregator(
		identity, s, const_multiplier);
	SG_REF(agg);

	// creating complex linear solver
	CConjugateOrthogonalCGSolver* solver=new CConjugateOrthogonalCGSolver();
	SG_REF(solver);

	CRationalApproximationIndividualJob* job
		=new CRationalApproximationIndividualJob(agg,
			(CLinearSolver<complex128_t, float64_t>*)solver, op, s, weight);
	SG_REF(job);
	job->compute();
	SG_UNREF(job);
	agg->finalize();
	CScalarResult<float64_t>* r=dynamic_cast<CScalarResult<float64_t>*>
		(agg->get_final_result());
	float64_t result=r->get_result();
	SG_UNREF(agg);
	SG_UNREF(op);
	SG_UNREF(identity);
	SG_UNREF(solver);

	EXPECT_NEAR(result, -0.73170731707317049, 1E-15);
}
#endif //HAVE_EIGEN3
