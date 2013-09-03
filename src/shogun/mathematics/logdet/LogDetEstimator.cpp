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
#include <shogun/lib/computation/job/ScalarResult.h>
#include <shogun/lib/computation/job/JobResultAggregator.h>
#include <shogun/mathematics/logdet/TraceSampler.h>
#include <shogun/mathematics/logdet/OperatorFunction.h>
#include <shogun/mathematics/logdet/LogDetEstimator.h>

namespace shogun
{

CLogDetEstimator::CLogDetEstimator()
	: CSGObject()
{
	init();

	SG_GCDEBUG("%s created (%p)\n", this->get_name(), this)
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

	SG_GCDEBUG("%s created (%p)\n", this->get_name(), this)
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

	SG_GCDEBUG("%s destroyed (%p)\n", this->get_name(), this)
}

SGVector<float64_t> CLogDetEstimator::sample(index_t num_estimates)
{
	REQUIRE(m_operator_log, "Operator function is NULL\n");
	// call the precompute of operator function to compute the prerequisites
	m_operator_log->precompute();

	REQUIRE(m_trace_sampler, "Trace sampler is NULL\n");
	// call the precompute of the sampler
	m_trace_sampler->precompute();

	REQUIRE(m_operator_log->get_operator()->get_dimension()\
		==m_trace_sampler->get_dimension(),
		"Dimension of the operator and sample doesn't match!\n");

	// for storing the aggregators that submit_jobs return
	CDynamicObjectArray* aggregators=new CDynamicObjectArray();
	index_t num_trace_samples=m_trace_sampler->get_num_samples();

	for (index_t i=0; i<num_estimates; ++i)
	{
		for (index_t j=0; j<num_trace_samples; ++j)
		{
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
	m_computation_engine->wait_for_all();

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

	SG_DEBUG("Leaving...\n")
	return samples;
}

}

