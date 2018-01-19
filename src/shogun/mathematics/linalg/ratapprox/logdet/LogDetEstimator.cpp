/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Soumyajit De
 */
#include <omp.h>
#include <shogun/lib/common.h>
#include <shogun/base/Parallel.h>
#include <shogun/base/progress.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/DynamicArray.h>
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

#ifdef HAVE_LAPACK
CLogDetEstimator::CLogDetEstimator(SGSparseMatrix<float64_t> sparse_mat)
	: CSGObject()
{
	init();

	CSparseMatrixOperator<float64_t>* op=
		new CSparseMatrixOperator<float64_t>(sparse_mat);

	float64_t accuracy=1E-5;

	CLanczosEigenSolver* eig_solver=new CLanczosEigenSolver(op);
	CCGMShiftedFamilySolver* linear_solver=new CCGMShiftedFamilySolver();

	m_operator_log=new CLogRationalApproximationCGM(op,eig_solver,linear_solver,accuracy);
	SG_REF(m_operator_log);

	#ifdef HAVE_COLPACK
	m_trace_sampler=new CProbingSampler(op,1,NATURAL,DISTANCE_TWO);
	#else
	m_trace_sampler=new CNormalSampler(op->get_dimension());
	#endif

	SG_REF(m_trace_sampler);

	SG_INFO("LogDetEstimator: %s with 1E-5 accuracy, %s as default\n", m_operator_log->get_name(),
		m_trace_sampler->get_name());
}
#endif //HAVE_LAPACK

CLogDetEstimator::CLogDetEstimator(CTraceSampler* trace_sampler,
	COperatorFunction<float64_t>* operator_log)
	: CSGObject()
{
	init();

	m_trace_sampler=trace_sampler;
	SG_REF(m_trace_sampler);

	m_operator_log=operator_log;
	SG_REF(m_operator_log);
}

void CLogDetEstimator::init()
{
	m_trace_sampler=NULL;
	m_operator_log=NULL;

	SG_ADD((CSGObject**)&m_trace_sampler, "trace_sampler",
		"Trace sampler for the log operator", MS_NOT_AVAILABLE);

	SG_ADD((CSGObject**)&m_operator_log, "operator_log",
		"The log operator function", MS_NOT_AVAILABLE);
}

CLogDetEstimator::~CLogDetEstimator()
{
	SG_UNREF(m_trace_sampler);
	SG_UNREF(m_operator_log);
}

CTraceSampler* CLogDetEstimator::get_trace_sampler(void) const
{
	SG_REF(m_trace_sampler);
	return m_trace_sampler;
}

COperatorFunction<float64_t>* CLogDetEstimator::get_operator_function(void) const
{
	SG_REF(m_operator_log);
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

// for storing the result
	CDynamicArray<float64_t> aggregators;
	index_t num_trace_samples=m_trace_sampler->get_num_samples();
//for omp
	int32_t num_vectors = num_estimates;
	auto pb = progress(range(num_vectors));
	int32_t num_threads;
	int64_t step;
#pragma omp parallel shared(num_threads, step)
			{

#ifdef HAVE_OPENMP //HAVE OPENMP
#pragma omp single
				{
					num_threads = omp_get_num_threads();
					step = num_vectors / num_threads;
					num_threads--;
				}
				int32_t thread_num = omp_get_thread_num();
#else
				num_threads = 0;
				step = num_vectors;
				int32_t thread_num = 0;
#endif
				int32_t start = thread_num * step;
				int32_t end = (thread_num == num_threads)
													? num_vectors
													: (thread_num + 1) * step;

#ifdef WIN32 //HAVE WIN32
				for (int32_t vec = start; vec < end; vec++)
#else
				for (int32_t vec = start; vec < end; vec++)
#endif
				{
					pb.print_progress();
					for (index_t j=0; j<num_trace_samples; ++j)
					{
						SG_INFO("Computing log-determinant trace sample %d/%d\n", j,
								num_trace_samples);
						// get the trace sampler vector
						SGVector<float64_t> s=m_trace_sampler->sample(j);
						// calculate the result for sample s
						float64_t agg=m_operator_log->solve(s);
#pragma omp critical //so that the dynamic array stays concurrent
						{
							aggregators.append_element(agg);
						}
					}
				}
			}
	// wait for all the computations to be completed
	SG_INFO("Waiting for computations to finish\n");
	#pragma omp barrier
	SG_INFO("All computations finished\n");

	// the samples vector which stores the estimates with averaging
	SGVector<float64_t> samples(num_estimates);
	samples.zero();

	//preparing the final result
	int32_t num_aggregates=aggregators.get_num_elements();
	index_t idx_row=0;
	index_t idx_col=0;
	for (int32_t i=0; i<num_aggregates; ++i)
	{
		samples[idx_col]+=aggregators.get_element(i);
		idx_row++;
		if (idx_row>=num_trace_samples)
		{
			idx_row=0;
			idx_col++;
		}
	}

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

	// for storing the result
	CDynamicArray<float64_t> aggregators;

	index_t num_trace_samples=m_trace_sampler->get_num_samples();

	int32_t num_vectors=num_estimates;
	auto pb = progress(range(num_vectors));
	int32_t num_threads;
	int64_t step;

#pragma omp parallel shared(num_threads, step)
				{

#ifdef HAVE_OPENMP //HAVE OPENMP
#pragma omp single
					{
						num_threads = omp_get_num_threads();
						step = num_vectors / num_threads;
						num_threads--;
					}
					int32_t thread_num = omp_get_thread_num();
#else
					num_threads = 0;
					step = num_vectors;
					int32_t thread_num = 0;
#endif
					int32_t start = thread_num * step;
					int32_t end = (thread_num == num_threads)
														? num_vectors
														: (thread_num + 1) * step;

#ifdef WIN32
					for (int32_t vec = start; vec < end; vec++)
#else
					for (int32_t vec = start; vec < end; vec++)
#endif
					{
						pb.print_progress();
						for (index_t j=0; j<num_trace_samples; ++j)
						{
							SG_INFO("Computing log-determinant trace sample %d/%d\n", j,
									num_trace_samples);
							// get the trace sampler vector
							SGVector<float64_t> s=m_trace_sampler->sample(j);
							// solve the result for s
							float64_t agg=m_operator_log->solve(s);
#pragma omp critical //aggregators array should be concurrent
							{
								aggregators.append_element(agg);
							}
						}
					}
				}
		// wait for all the computations to be completed
		SG_INFO("Waiting for computations to finish\n");
#pragma omp barrier
		SG_INFO("All computations finished\n");

	// the samples matrix which stores the estimates without averaging
	// dimension: number of trace samples x number of log-det estimates
	SGMatrix<float64_t> samples(num_trace_samples, num_estimates);

	int32_t num_aggregates=aggregators.get_num_elements();
	for (int32_t i=0; i<num_aggregates; ++i)
	{
		index_t idx_row=i%num_trace_samples;
		index_t idx_col=i/num_trace_samples;
		samples(idx_row, idx_col)=aggregators.get_element(i);
	}

	SG_DEBUG("Leaving\n")
	return samples;
}

}
