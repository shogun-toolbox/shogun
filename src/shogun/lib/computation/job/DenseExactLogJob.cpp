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
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/computation/job/ScalarResult.h>
#include <shogun/mathematics/logdet/DenseMatrixOperator.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/lib/computation/job/DenseExactLogJob.h>

using namespace Eigen;

namespace shogun
{

CDenseExactLogJob::CDenseExactLogJob()
	: CIndependentJob(), m_log_operator(NULL)
{
	SG_GCDEBUG("%s created (%p)\n", this->get_name(), this)
}

CDenseExactLogJob::CDenseExactLogJob(CJobResultAggregator* aggregator,
	CDenseMatrixOperator<float64_t>* log_operator, SGVector<float64_t> vector)
	: CIndependentJob(aggregator), m_log_operator(log_operator), m_vector(vector)
{
	SG_REF(m_log_operator);
	SG_GCDEBUG("%s created (%p)\n", this->get_name(), this)
}

CDenseExactLogJob::~CDenseExactLogJob()
{
	SG_UNREF(m_log_operator);
	SG_GCDEBUG("%s destroyed (%p)\n", this->get_name(), this)
}

void CDenseExactLogJob::compute()
{
	SG_DEBUG("Entering...\n")

	// apply the log to m_vector
	SGVector<float64_t> vec=m_log_operator->apply(m_vector);

	// compute the vector-vector dot product using Eigen3
	Map<VectorXd> v(vec.vector, vec.vlen);
	Map<VectorXd> s(m_vector.vector, m_vector.vlen);
	
	CScalarResult<float64_t>* result=new CScalarResult<float64_t>(s.dot(v));
	SG_REF(result);
	m_aggregator->submit_result(result);
	SG_UNREF(result);

	SG_DEBUG("Leaving...\n")
}

SGVector<float64_t> CDenseExactLogJob::get_vector() const
{
	return m_vector;
}

CDenseMatrixOperator<float64_t>* CDenseExactLogJob::get_operator() const
{
	return m_log_operator;
}

}
#endif // HAVE_EIGEN3
