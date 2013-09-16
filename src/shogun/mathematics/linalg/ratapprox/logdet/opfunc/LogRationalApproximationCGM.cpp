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
#include <shogun/base/Parameter.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/linalg/linsolver/CGMShiftedFamilySolver.h>
#include <shogun/mathematics/linalg/linop/LinearOperator.h>
#include <shogun/mathematics/linalg/ratapprox/logdet/opfunc/LogRationalApproximationCGM.h>
#include <shogun/mathematics/linalg/ratapprox/logdet/computation/job/RationalApproximationCGMJob.h>
#include <shogun/lib/computation/aggregator/StoreScalarAggregator.h>
#include <shogun/lib/computation/engine/IndependentComputationEngine.h>

using namespace Eigen;

namespace shogun
{

CLogRationalApproximationCGM::CLogRationalApproximationCGM()
	: CRationalApproximation(NULL, NULL, NULL, 0, OF_LOG)
{
	init();
}

CLogRationalApproximationCGM::CLogRationalApproximationCGM(
	CLinearOperator<float64_t>* linear_operator,
	CIndependentComputationEngine* computation_engine,
	CEigenSolver* eigen_solver,
	CCGMShiftedFamilySolver* linear_solver,
	float64_t desired_accuracy)
	: CRationalApproximation(linear_operator, computation_engine,
	  eigen_solver, desired_accuracy, OF_LOG)
{
	init();

	m_linear_solver=linear_solver;
	SG_REF(m_linear_solver);
}

void CLogRationalApproximationCGM::init()
{
	m_linear_solver=NULL;

	SG_ADD((CSGObject**)&m_linear_solver, "linear_solver",
		"Linear solver for complex systems", MS_NOT_AVAILABLE);

	SG_ADD(&m_negated_shifts, "negated_shifts",
		"Negated shifts", MS_NOT_AVAILABLE);
}

CLogRationalApproximationCGM::~CLogRationalApproximationCGM()
{
	SG_UNREF(m_linear_solver);
}

CJobResultAggregator* CLogRationalApproximationCGM::submit_jobs(
	SGVector<float64_t> sample)
{
	SG_DEBUG("Entering\n");
	REQUIRE(sample.vector, "Sample is not initialized!\n");
	REQUIRE(m_linear_operator, "Operator is not initialized!\n");
	REQUIRE(m_computation_engine, "Computation engine is NULL\n");

	// create the scalar aggregator
	CStoreScalarAggregator<float64_t>* agg=new CStoreScalarAggregator<float64_t>();
	// we don't want the aggregator to be destroyed when the job is unref-ed
	SG_REF(agg);

	// we need to take the negation of the shifts for this case
	if (m_negated_shifts.vector==NULL)
	{
		m_negated_shifts=SGVector<complex64_t>(m_shifts.vlen);
		Map<VectorXcd> shifts(m_shifts.vector, m_shifts.vlen);
		Map<VectorXcd> negated_shifts(m_negated_shifts.vector, m_negated_shifts.vlen);
		negated_shifts=-shifts;
	}

	// create one CG-M job for current sample vector which solves for all
	// the shifts, and computes the final result and stores that in the aggregator
	CRationalApproximationCGMJob* job
			=new CRationalApproximationCGMJob(agg, m_linear_solver,
			m_linear_operator, sample, m_negated_shifts, m_weights, m_constant_multiplier);
	SG_REF(job);

	m_computation_engine->submit_job(job);

	// we can safely unref the job here, computation engine takes it from here
	SG_UNREF(job);
	
	SG_DEBUG("Leaving\n");
	return agg;
}

}
#endif // HAVE_EIGEN3
