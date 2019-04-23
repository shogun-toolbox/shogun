/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sunil Mahendrakar, Heiko Strathmann, Soumyajit De, Bjoern Esser
 */
#include <shogun/base/Parallel.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/common.h>
#include <shogun/mathematics/linalg/eigsolver/LanczosEigenSolver.h>
#include <shogun/mathematics/linalg/linop/DenseMatrixOperator.h>
#include <shogun/mathematics/linalg/linop/SparseMatrixOperator.h>
#include <shogun/mathematics/linalg/linsolver/CGMShiftedFamilySolver.h>
#include <shogun/mathematics/linalg/ratapprox/logdet/LogDetEstimator.h>
#include <shogun/mathematics/linalg/ratapprox/logdet/opfunc/DenseMatrixExactLog.h>
#include <shogun/mathematics/linalg/ratapprox/logdet/opfunc/LogRationalApproximationCGM.h>
#include <shogun/mathematics/linalg/ratapprox/opfunc/OperatorFunction.h>
#include <shogun/mathematics/linalg/ratapprox/tracesampler/NormalSampler.h>
#include <shogun/mathematics/linalg/ratapprox/tracesampler/ProbingSampler.h>
#include <shogun/mathematics/linalg/ratapprox/tracesampler/TraceSampler.h>

namespace shogun
{

LogDetEstimator::LogDetEstimator()
	: SGObject()
{
	init();
}

#ifdef HAVE_LAPACK
LogDetEstimator::LogDetEstimator(SGSparseMatrix<float64_t> sparse_mat)
	: SGObject()
{
	init();

	auto op=
		std::make_shared<SparseMatrixOperator<float64_t>>(sparse_mat);

	float64_t accuracy=1E-5;

	auto eig_solver=std::make_shared<LanczosEigenSolver>(op);
	auto linear_solver=std::make_shared<CGMShiftedFamilySolver>();

	m_operator_log = std::make_shared<LogRationalApproximationCGM>(
		op, eig_solver, linear_solver, accuracy);


	#ifdef HAVE_COLPACK
	m_trace_sampler=std::make_shared<ProbingSampler>(op,1,NATURAL,DISTANCE_TWO);
	#else
	m_trace_sampler=std::make_shared<NormalSampler>(op->get_dimension());
	#endif



	SG_INFO(
		"LogDetEstimator: %s with 1E-5 accuracy, %s as default\n",
		m_operator_log->get_name(), m_trace_sampler->get_name());
}
#endif //HAVE_LAPACK

LogDetEstimator::LogDetEstimator(
	std::shared_ptr<TraceSampler> trace_sampler, std::shared_ptr<OperatorFunction<float64_t>> operator_log)
	: SGObject()
{
	init();

	m_trace_sampler=trace_sampler;


	m_operator_log=operator_log;

}

void LogDetEstimator::init()
{
	m_trace_sampler=NULL;
	m_operator_log=NULL;

	SG_ADD((std::shared_ptr<SGObject>*)&m_trace_sampler, "trace_sampler",
		"Trace sampler for the log operator");

	SG_ADD((std::shared_ptr<SGObject>*)&m_operator_log, "operator_log",
		"The log operator function");
}

LogDetEstimator::~LogDetEstimator()
{


}

std::shared_ptr<TraceSampler> LogDetEstimator::get_trace_sampler(void) const
{

	return m_trace_sampler;
}

std::shared_ptr<OperatorFunction<float64_t>> LogDetEstimator::get_operator_function(void) const
{

	return m_operator_log;
}

SGVector<float64_t> LogDetEstimator::sample(index_t num_estimates)
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

	index_t num_trace_samples=m_trace_sampler->get_num_samples();
	SGVector<float64_t> samples(num_estimates);
	samples.zero();
	float64_t result = 0.0;

#pragma omp parallel for reduction(+ : result)
	for (index_t i = 0; i < num_estimates; ++i)
	{
		result = 0.0;
		for (index_t j = 0; j < num_trace_samples; ++j)
		{
			SG_INFO(
				"Computing log-determinant trace sample %d/%d\n", j,
				num_trace_samples);
			// get the trace sampler vector
			SGVector<float64_t> s = m_trace_sampler->sample(j);
			// calculate the result for sample s and add it to previous
			result += m_operator_log->compute(s);
		}
		samples[i] = result;
	}

	SG_INFO("Finished computing %d log-det estimates\n", num_estimates);

	SG_DEBUG("Leaving\n");
	return samples;
}

SGMatrix<float64_t> LogDetEstimator::sample_without_averaging(
	index_t num_estimates)
{
	SG_DEBUG("Entering...\n")

	REQUIRE(m_operator_log, "Operator function is NULL\n");
	// call the precompute of operator function to compute all prerequisites
	m_operator_log->precompute();

	REQUIRE(m_trace_sampler, "Trace sampler is NULL\n");
	// call the precompute of the sampler
	m_trace_sampler->precompute();

	index_t num_trace_samples = m_trace_sampler->get_num_samples();
	SGMatrix<float64_t> samples(num_trace_samples, num_estimates);

#pragma omp parallel for
	for (index_t i = 0; i < num_estimates; ++i)
	{
		for (index_t j = 0; j < num_trace_samples; ++j)
		{
			SG_INFO(
				"Computing log-determinant trace sample %d/%d\n", j,
				num_trace_samples);
			// get the trace sampler vector
			SGVector<float64_t> s = m_trace_sampler->sample(j);
			// solve the result for s
			float64_t result = m_operator_log->compute(s);
			{
				samples(i, j) = result;
			}
		}
	}

	SG_DEBUG("Leaving\n")
	return samples;
}

}
