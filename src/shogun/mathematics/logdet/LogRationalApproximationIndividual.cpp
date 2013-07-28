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
#include <shogun/mathematics/logdet/LinearSolver.h>
#include <shogun/mathematics/logdet/DenseMatrixOperator.h>
#include <shogun/mathematics/logdet/LogRationalApproximationIndividual.h>
#include <shogun/lib/computation/job/RationalApproximationIndividualJob.h>
#include <shogun/lib/computation/job/IndividualJobResultAggregator.h>
#include <shogun/lib/computation/engine/IndependentComputationEngine.h>

namespace shogun
{

CLogRationalApproximationIndividual::CLogRationalApproximationIndividual()
	: CRationalApproximation(NULL, NULL, NULL, 0, OF_LOG)
{
	init();

	SG_GCDEBUG("%s created (%p)\n", this->get_name(), this)
}

CLogRationalApproximationIndividual::CLogRationalApproximationIndividual(
	CDenseMatrixOperator<float64_t, float64_t>* linear_operator,
	CIndependentComputationEngine* computation_engine,
	CEigenSolver* eigen_solver, 
	CLinearSolver<complex64_t, float64_t>* linear_solver,
	index_t num_shifts)
	: CRationalApproximation(linear_operator, computation_engine,
	  eigen_solver, num_shifts, OF_LOG)
{
	init();

	m_linear_solver=linear_solver;
	SG_REF(m_linear_solver);
	
	SG_GCDEBUG("%s created (%p)\n", this->get_name(), this)
}

void CLogRationalApproximationIndividual::init()
{
	m_linear_solver=NULL;

	SG_ADD((CSGObject**)&m_linear_solver, "linear_solver",
		"Direct linear solver for complex systems", MS_NOT_AVAILABLE);
}

CLogRationalApproximationIndividual::~CLogRationalApproximationIndividual()
{
	SG_UNREF(m_linear_solver);

	SG_GCDEBUG("%s destroyed (%p)\n", this->get_name(), this)
}

CJobResultAggregator* CLogRationalApproximationIndividual::submit_jobs(
	SGVector<float64_t> sample)
{
	SG_DEBUG("OperatorFunction::submit_jobs(): Entering..\n");
	REQUIRE(sample.vector, "Sample is not initialized!\n");
	REQUIRE(m_linear_operator, "Operator is not initialized!\n");
	REQUIRE(m_computation_engine, "Computation engine is NULL\n");

	// create the aggregator with sample, and the multiplier
	CIndividualJobResultAggregator* agg=new CIndividualJobResultAggregator(
		m_linear_operator, sample, m_constant_multiplier);
	// we don't want the aggregator to be destroyed when the job is unref-ed
	SG_REF(agg);

	// create a complex copy of the dense matrix linear operator
	SGMatrix<float64_t> m=dynamic_cast<CDenseMatrixOperator<float64_t>*>
		(m_linear_operator)->get_matrix_operator();

	REQUIRE(m.matrix, "Matrix is not initialized!\n");
	
	SGMatrix<complex64_t> complex_m(m.num_rows, m.num_cols);
	for (index_t i=0; i<m.num_cols; ++i)
	{
		for (index_t j=0; j<m.num_rows; ++j)
			complex_m(j,i)=complex64_t(m(j,i));
	}
	CDenseMatrixOperator<complex64_t> complex_op(complex_m);

	// create num_shifts number of jobs for current sample vector
	for (index_t i=0; i<m_num_shifts; ++i)
	{
		// create a deep copy of the operator
		CDenseMatrixOperator<complex64_t, complex64_t>* shifted_op
			=new CDenseMatrixOperator<complex64_t>(complex_op);

		// move the shift inside the operator
		// (see CRationalApproximation)
		SGVector<complex64_t> diag=shifted_op->get_diagonal();
		for (index_t j=0; j<diag.vlen; ++j)
			diag[j]-=m_shifts[i];
		shifted_op->set_diagonal(diag);

		// create a job and submit to the engine
		CRationalApproximationIndividualJob* job
			=new CRationalApproximationIndividualJob(agg, m_linear_solver, 
				shifted_op, sample, m_weights[i]);
		SG_REF(job);

		m_computation_engine->submit_job(job);

		// we can safely unref the job here, computation engine takes it from here
		SG_UNREF(job);
	}

	SG_DEBUG("OperatorFunction::submit_jobs(): Leaving..\n");
	return agg;
}

}
#endif // HAVE_EIGEN3
