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

	m_operator_log = new CLogRationalApproximationCGM(
		op, eig_solver, linear_solver, accuracy);
	SG_REF(m_operator_log);

	#ifdef HAVE_COLPACK
	m_trace_sampler=new CProbingSampler(op,1,NATURAL,DISTANCE_TWO);
	#else
	m_trace_sampler=new CNormalSampler(op->get_dimension());
	#endif

	SG_REF(m_trace_sampler);

	SG_INFO(
		"LogDetEstimator: %s with 1E-5 accuracy, %s as default\n",
		m_operator_log->get_name(), m_trace_sampler->get_name());
}
#endif //HAVE_LAPACK

CLogDetEstimator::CLogDetEstimator(
	CTraceSampler* trace_sampler, COperatorFunction<float64_t>* operator_log)
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
