/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Soumyajit De
 */

#include <shogun/lib/common.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/DynamicObjectArray.h>
#include <shogun/lib/computation/engine/IndependentComputationEngine.h>
#include <shogun/lib/computation/engine/SerialComputationEngine.h>
#include <shogun/lib/computation/jobresult/ScalarResult.h>
#include <shogun/lib/computation/aggregator/JobResultAggregator.h>
#include <shogun/mathematics/linalg/linop/DenseMatrixOperator.h>
#include <shogun/mathematics/linalg/linop/SparseMatrixOperator.h>
#include <shogun/mathematics/linalg/eigsolver/LanczosEigenSolver.h>
#include <shogun/mathematics/linalg/linsolver/CGMShiftedFamilySolver.h>
#include <shogun/mathematics/linalg/ratapprox/tracesampler/TraceSampler.h>
#include <shogun/mathematics/linalg/ratapprox/tracesampler/ProbingSampler.h>
#include <shogun/mathematics/linalg/ratapprox/tracesampler/NormalSampler.h>
#include <shogun/mathematics/linalg/ratapprox/opfunc/OperatorFunction.h>
#include <shogun/mathematics/linalg/ratapprox/logdet/LogDetEstimator.h>
#include <shogun/mathematics/linalg/ratapprox/logdet/opfunc/DenseMatrixExactLog.h>
#include <shogun/mathematics/linalg/ratapprox/logdet/opfunc/LogRationalApproximationCGM.h>

namespace shogun
{

CLogDetEstimator::CLogDetEstimator()
	: CSGObject()
{
	init();
}

CLogDetEstimator::CLogDetEstimator(SGMatrix<float64_t> dense_mat) 
	: CSGObject()
{
	init();

	m_computation_engine = new CSerialComputationEngine();
	SG_REF(m_computation_engine);

	CDenseMatrixOperator<float64_t>* op = new CDenseMatrixOperator<float64_t>(dense_mat);
	
	m_operator_log = new CDenseMatrixExactLog(op,m_computation_engine);
	SG_REF(m_operator_log);

	m_trace_sampler = new CNormalSampler(dense_mat.num_rows);
	SG_REF(m_trace_sampler);

	SG_INFO("Using CSerialComputationEngine, CDenseMatrixExactLog, CNormalSampler as default\n");
}

CLogDetEstimator::CLogDetEstimator(SGSparseMatrix<float64_t> sparse_mat) 
	: CSGObject()
{
	init();

	m_computation_engine = new CSerialComputationEngine();
	SG_REF(m_computation_engine);

	CSparseMatrixOperator<float64_t>* op = new CSparseMatrixOperator<float64_t>(sparse_mat);
	float64_t accuracy=1E-5;
	CLanczosEigenSolver* eig_solver = new CLanczosEigenSolver(op);
	CCGMShiftedFamilySolver* linear_solver = new CCGMShiftedFamilySolver();

	m_operator_log=new CLogRationalApproximationCGM(op,m_computation_engine,
		eig_solver,linear_solver,accuracy);
	SG_REF(m_operator_log);

	#ifdef HAVE_LOPACK
	m_trace_sampler=new CProbingSampler(op,1,NATURAL,DISTANCE_TWO);
	SG_REF(m_trace_sampler);

	SG_INFO("CLogDetEstimator:using CSerialComputationEngine, CLogRationalApproximationCGM, CProbingSampler as default\n");

	#else
	m_trace_sampler = new CNormalSampler(op->get_dimension());

	SG_INFO("CLogDetEstimator:using CSerialComputationEngine, CLogRationalApproximationCGM, CNormalSampler as default\n");
	#endif
}

CLogDetEstimator::CLogDetEstimator(CTraceSampler* trace_sampler,
	COperatorFunction<float64_t>* operator_log,
	CIndependentComputationEngine* computation_engine)
	: CSGObject()
{
	init();

	m_trace_sampler=trace_sampler;
	SG_REF(m_trace_sampler);

	m_operator_log=operator_log;
	SG_REF(m_operator_log);

	m_computation_engine=computation_engine;
	SG_REF(m_computation_engine);
}

void CLogDetEstimator::init()
{
	m_trace_sampler=NULL;
	m_operator_log=NULL;
	m_computation_engine=NULL;

	SG_ADD((CSGObject**)&m_trace_sampler, "trace_sampler",
		"Trace sampler for the log operator", MS_NOT_AVAILABLE);

	SG_ADD((CSGObject**)&m_operator_log, "operator_log",
		"The log operator function", MS_NOT_AVAILABLE);

	SG_ADD((CSGObject**)&m_computation_engine, "computation_engine",
		"The computation engine for the jobs", MS_NOT_AVAILABLE);
}

CLogDetEstimator::~CLogDetEstimator()
{
	SG_UNREF(m_trace_sampler);
	SG_UNREF(m_operator_log);
	SG_UNREF(m_computation_engine);
}

CTraceSampler* CLogDetEstimator::get_trace_sampler(void) const
{
	return m_trace_sampler;
}

CIndependentComputationEngine* CLogDetEstimator::get_computation_engine(void) const
{
	return m_computation_engine;
}

COperatorFunction<float64_t>* CLogDetEstimator::get_operator_function(void) const
{
	return m_operator_log;
}

SGVector<float64_t> CLogDetEstimator::sample(index_t num_estimates)
{
	SG_DEBUG("Entering\n");
	SG_INFO("Computing %d log-det estimates\n", num_estimates);

	REQUIRE(m_operator_log, "Operator function is NULL\n");
	// call the precompute of operator function to compute the prerequisites
	m_operator_log->precompute();

	REQUIRE(m_trace_sampler, "Trace sampler is NULL\n");
	// call the precompute of the sampler
	m_trace_sampler->precompute();

	REQUIRE(m_operator_log->get_operator()->get_dimension()\
		==m_trace_sampler->get_dimension(),
		"Mismatch in dimensions of the operator and trace-sampler, %d vs %d!\n",
		m_operator_log->get_operator()->get_dimension(),
		m_trace_sampler->get_dimension());

	// for storing the aggregators that submit_jobs return
	CDynamicObjectArray* aggregators=new CDynamicObjectArray();
	index_t num_trace_samples=m_trace_sampler->get_num_samples();

	for (index_t i=0; i<num_estimates; ++i)
	{
		for (index_t j=0; j<num_trace_samples; ++j)
		{
			SG_INFO("Computing log-determinant trace sample %d/%d\n", j,
					num_trace_samples);

			SG_DEBUG("Creating job for estimate %d, trace sample %d/%d\n", i, j,
					num_trace_samples);
			// get the trace sampler vector
			SGVector<float64_t> s=m_trace_sampler->sample(j);
			// create jobs with the sample vector and store the aggregator
			CJobResultAggregator* agg=m_operator_log->submit_jobs(s);
			aggregators->append_element(agg);
			SG_UNREF(agg);
		}
	}

	REQUIRE(m_computation_engine, "Computation engine is NULL\n");

	// wait for all the jobs to be completed
	SG_INFO("Waiting for jobs to finish\n");
	m_computation_engine->wait_for_all();
	SG_INFO("All jobs finished, aggregating results\n");

	// the samples vector which stores the estimates with averaging
	SGVector<float64_t> samples(num_estimates);
	samples.zero();

	// use the aggregators to find the final result
	// use the same order as job submission to combine results
	int32_t num_aggregates=aggregators->get_num_elements();
	index_t idx_row=0;
	index_t idx_col=0;
	for (int32_t i=0; i<num_aggregates; ++i)
	{
		// this cast is safe due to above way of building the array
		CJobResultAggregator* agg=dynamic_cast<CJobResultAggregator*>
			(aggregators->get_element(i));
		ASSERT(agg);

		// call finalize on all the aggregators, cast is safe again
		agg->finalize();
		CScalarResult<float64_t>* r=dynamic_cast<CScalarResult<float64_t>*>
			(agg->get_final_result());
		ASSERT(r);

		// iterate through indices, group results in the same way as jobs
		samples[idx_col]+=r->get_result();
		idx_row++;
		if (idx_row>=num_trace_samples)
		{
			idx_row=0;
			idx_col++;
		}

		SG_UNREF(agg);
	}

	// clear all aggregators
	SG_UNREF(aggregators)

	SG_INFO("Finished computing %d log-det estimates\n", num_estimates);

	SG_DEBUG("Leaving\n");
	return samples;
}

SGMatrix<float64_t> CLogDetEstimator::sample_without_averaging(
	index_t num_estimates)
{
	SG_DEBUG("Entering...\n")

	REQUIRE(m_operator_log, "Operator function is NULL\n");
	// call the precompute of operator function to compute all prerequisites
	m_operator_log->precompute();

	REQUIRE(m_trace_sampler, "Trace sampler is NULL\n");
	// call the precompute of the sampler
	m_trace_sampler->precompute();

	// for storing the aggregators that submit_jobs return
	CDynamicObjectArray aggregators;
	index_t num_trace_samples=m_trace_sampler->get_num_samples();

	for (index_t i=0; i<num_estimates; ++i)
	{
		for (index_t j=0; j<num_trace_samples; ++j)
		{
			// get the trace sampler vector
			SGVector<float64_t> s=m_trace_sampler->sample(j);
			// create jobs with the sample vector and store the aggregator
			CJobResultAggregator* agg=m_operator_log->submit_jobs(s);
			aggregators.append_element(agg);
			SG_UNREF(agg);
		}
	}

	REQUIRE(m_computation_engine, "Computation engine is NULL\n");
	// wait for all the jobs to be completed
	m_computation_engine->wait_for_all();

	// the samples matrix which stores the estimates without averaging
	// dimension: number of trace samples x number of log-det estimates
	SGMatrix<float64_t> samples(num_trace_samples, num_estimates);

	// use the aggregators to find the final result
	int32_t num_aggregates=aggregators.get_num_elements();
	for (int32_t i=0; i<num_aggregates; ++i)
	{
		CJobResultAggregator* agg=dynamic_cast<CJobResultAggregator*>
			(aggregators.get_element(i));
		if (!agg)
			SG_ERROR("Element is not CJobResultAggregator type!\n");

		// call finalize on all the aggregators
		agg->finalize();
		CScalarResult<float64_t>* r=dynamic_cast<CScalarResult<float64_t>*>
			(agg->get_final_result());
		if (!r)
			SG_ERROR("Result is not CScalarResult type!\n");

		// its important that we don't just unref the result here
		index_t idx_row=i%num_trace_samples;
		index_t idx_col=i/num_trace_samples;
		samples(idx_row, idx_col)=r->get_result();
		SG_UNREF(agg);
	}

	// clear all aggregators
	aggregators.clear_array();

	SG_DEBUG("Leaving\n")
	return samples;
}

}

