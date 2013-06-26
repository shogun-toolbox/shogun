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

#if EIGEN_VERSION_AT_LEAST(3,1,0)
#include <unsupported/Eigen/MatrixFunctions>
#endif // EIGEN_VERSION_AT_LEAST(3,1,0)

#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/logdet/DenseMatrixExactLog.h>
#include <shogun/mathematics/logdet/DenseMatrixOperator.h>
#include <shogun/lib/computation/job/DenseExactLogJob.h>
#include <shogun/lib/computation/job/StoreScalarAggregator.h>
#include <shogun/lib/computation/engine/IndependentComputationEngine.h>

using namespace Eigen;

namespace shogun
{

CDenseMatrixExactLog::CDenseMatrixExactLog()
	: COperatorFunction<float64_t>(NULL, NULL, OF_LOG)
{
	SG_GCDEBUG("%s created (%p)\n", this->get_name(), this)
}

CDenseMatrixExactLog::CDenseMatrixExactLog(
	CDenseMatrixOperator<float64_t>* op,
	CIndependentComputationEngine* engine)
	: COperatorFunction<float64_t>(
		(CLinearOperator<float64_t>*)op, engine, OF_LOG)
{
	SG_GCDEBUG("%s created (%p)\n", this->get_name(), this)
}

CDenseMatrixExactLog::~CDenseMatrixExactLog()
{
	SG_GCDEBUG("%s destroyed (%p)\n", this->get_name(), this)
}

#if EIGEN_VERSION_AT_LEAST(3,1,0)
void CDenseMatrixExactLog::precompute()
{
	SG_DEBUG("Entering...\n");

	// check for proper downcast
	CDenseMatrixOperator<float64_t>* op
		=dynamic_cast<CDenseMatrixOperator<float64_t>*>(m_linear_operator);
	REQUIRE(op, "Operator not an instance of DenseMatrixOperator!\n");
	SGMatrix<float64_t> m=op->get_matrix_operator();

	// compute log(C) using Eigen3
	Map<MatrixXd> mat(m.matrix, m.num_rows, m.num_cols);
	SGMatrix<float64_t> log_m(m.num_rows, m.num_cols);
	Map<MatrixXd> log_mat(log_m.matrix, log_m.num_rows, log_m.num_cols);
	log_mat=mat.log();

	// the log(C) is also a linear operator here
	// reset the operator of this function with log(C)
	SG_UNREF(m_linear_operator);
	m_linear_operator=new CDenseMatrixOperator<float64_t>(log_m);
	SG_REF(m_linear_operator);

	SG_DEBUG("Leaving...\n");
}
#else
void CDenseMatrixExactLog::precompute()
{
	SG_WARNING("Eigen3.1.0 or later required!\n")
}
#endif // EIGEN_VERSION_AT_LEAST(3,1,0)

CJobResultAggregator* CDenseMatrixExactLog::submit_jobs(SGVector<float64_t>
	sample)
{
	SG_DEBUG("Entering...\n");

	CStoreScalarAggregator<float64_t>* agg=new CStoreScalarAggregator<float64_t>;
	// we don't want the aggregator to be destroyed when the job is unref-ed
	SG_REF(agg);
	CDenseExactLogJob* job=new CDenseExactLogJob(agg, 
		dynamic_cast<CDenseMatrixOperator<float64_t>*>(m_linear_operator), sample);
	SG_REF(job);
	// sanity check
	REQUIRE(m_computation_engine, "Computation engine is NULL\n");
	m_computation_engine->submit_job(job);
	// we can safely unref the job here, computation engine takes it from here
	SG_UNREF(job);

	SG_DEBUG("Leaving...\n");
	return agg;
}

}
#endif // HAVE_EIGEN3
