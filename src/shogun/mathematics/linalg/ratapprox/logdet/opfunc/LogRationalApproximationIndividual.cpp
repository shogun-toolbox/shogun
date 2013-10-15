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
#include <shogun/mathematics/linalg/linsolver/LinearSolver.h>
#include <shogun/mathematics/linalg/linop/DenseMatrixOperator.h>
#include <shogun/mathematics/linalg/linop/SparseMatrixOperator.h>
#include <shogun/mathematics/linalg/ratapprox/logdet/opfunc/LogRationalApproximationIndividual.h>
#include <shogun/mathematics/linalg/ratapprox/logdet/computation/job/RationalApproximationIndividualJob.h>
#include <shogun/mathematics/linalg/ratapprox/logdet/computation/aggregator/IndividualJobResultAggregator.h>
#include <shogun/lib/computation/engine/IndependentComputationEngine.h>
#include <typeinfo>

namespace shogun
{

CLogRationalApproximationIndividual::CLogRationalApproximationIndividual()
	: CRationalApproximation(NULL, NULL, NULL, 0, OF_LOG)
{
	init();

	SG_GCDEBUG("%s created (%p)\n", this->get_name(), this)
}

CLogRationalApproximationIndividual::CLogRationalApproximationIndividual(
	CMatrixOperator<float64_t>* linear_operator,
	CIndependentComputationEngine* computation_engine,
	CEigenSolver* eigen_solver,
	CLinearSolver<complex128_t, float64_t>* linear_solver,
	float64_t desired_accuracy)
	: CRationalApproximation(linear_operator, computation_engine,
	  eigen_solver, desired_accuracy, OF_LOG)
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
		"Linear solver for complex systems", MS_NOT_AVAILABLE);
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

	// this enum will save from repeated typechecking for all jobs
	enum typeID {DENSE=1, SPARSE, UNKNOWN} operator_type=UNKNOWN;

	// create a complex copy of the matrix linear operator
	CMatrixOperator<complex128_t>* complex_op=NULL;
	if (typeid(*m_linear_operator)==typeid(CDenseMatrixOperator<float64_t>))
	{
		operator_type=DENSE;

		CDenseMatrixOperator<float64_t>* op
			=dynamic_cast<CDenseMatrixOperator<float64_t>*>(m_linear_operator);

		REQUIRE(op->get_matrix_operator().matrix, "Matrix is not initialized!\n");

		// create complex dense matrix operator
		complex_op=static_cast<CDenseMatrixOperator<complex128_t>*>(*op);
	}
	else if (typeid(*m_linear_operator)==typeid(CSparseMatrixOperator<float64_t>))
	{
		operator_type=SPARSE;

		CSparseMatrixOperator<float64_t>* op
			=dynamic_cast<CSparseMatrixOperator<float64_t>*>(m_linear_operator);

		REQUIRE(op->get_matrix_operator().sparse_matrix, "Matrix is not initialized!\n");

		// create complex sparse matrix operator
		complex_op=static_cast<CSparseMatrixOperator<complex128_t>*>(*op);
	}
	else
	{
		// something weird happened
		SG_ERROR("OperatorFunction::submit_jobs(): Unknown MatrixOperator given!\n");
	}

	// create num_shifts number of jobs for current sample vector
	for (index_t i=0; i<m_num_shifts; ++i)
	{
		// create a deep copy of the operator
		CMatrixOperator<complex128_t>* shifted_op=NULL;

		switch(operator_type)
		{
		case DENSE:
			shifted_op=new CDenseMatrixOperator<complex128_t>
				(*dynamic_cast<CDenseMatrixOperator<complex128_t>*>(complex_op));
			break;
		case SPARSE:
			shifted_op=new CSparseMatrixOperator<complex128_t>
				(*dynamic_cast<CSparseMatrixOperator<complex128_t>*>(complex_op));
			break;
		default:
			break;
		}

		REQUIRE(shifted_op, "OperatorFunction::submit_jobs():"
			"MatrixOperator typeinfo was not detected!\n");

		// move the shift inside the operator
		// (see CRationalApproximation)
		SGVector<complex128_t> diag=shifted_op->get_diagonal();
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

	SG_UNREF(complex_op);

	SG_DEBUG("OperatorFunction::submit_jobs(): Leaving..\n");
	return agg;
}

}
#endif // HAVE_EIGEN3
