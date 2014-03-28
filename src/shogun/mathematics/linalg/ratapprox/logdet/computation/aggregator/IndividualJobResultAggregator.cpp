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
#include <shogun/lib/SGVector.h>
#include <shogun/lib/computation/jobresult/ScalarResult.h>
#include <shogun/base/Parameter.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/linalg/linop/LinearOperator.h>
#include <shogun/mathematics/linalg/ratapprox/logdet/computation/aggregator/IndividualJobResultAggregator.h>

using namespace Eigen;

namespace shogun
{
CIndividualJobResultAggregator::CIndividualJobResultAggregator()
	: CStoreVectorAggregator<complex128_t>(),
	  m_const_multiplier(0.0)
{
	init();

	SG_GCDEBUG("%s created (%p)\n", this->get_name(), this)
}

CIndividualJobResultAggregator::CIndividualJobResultAggregator(
	CLinearOperator< SGVector<float64_t>, SGVector<float64_t> >* linear_operator,
	SGVector<float64_t> vector,
	const float64_t& const_multiplier)
	: CStoreVectorAggregator<complex128_t>(vector.vlen),
	  m_const_multiplier(const_multiplier)
{
	init();

	m_vector=vector;

	m_linear_operator=linear_operator;
	SG_REF(m_linear_operator);

	SG_GCDEBUG("%s created (%p)\n", this->get_name(), this)
}

void CIndividualJobResultAggregator::init()
{
	m_linear_operator=NULL;

	SG_ADD(&m_vector, "sample_vector",
		"The sample vector to perform final dot product", MS_NOT_AVAILABLE);

	SG_ADD((CSGObject**)&m_linear_operator, "linear_operator",
		"The linear operator to apply on the aggregation", MS_NOT_AVAILABLE);
}

CIndividualJobResultAggregator::~CIndividualJobResultAggregator()
{
	SG_UNREF(m_linear_operator);

	SG_GCDEBUG("%s destroyed (%p)\n", this->get_name(), this)
}

void CIndividualJobResultAggregator::finalize()
{
	// take out the imaginary part of the aggegation before
	// applying linear operator
	SGVector<float64_t> imag_agg=m_aggregate.get_imag();
	SGVector<float64_t> agg=m_linear_operator->apply(imag_agg);

	// perform dot product
	Map<VectorXd> map_agg(agg.vector, agg.vlen);
	Map<VectorXd> map_vector(m_vector.vector, m_vector.vlen);
	float64_t result=map_vector.dot(map_agg);

	result*=m_const_multiplier;

	// form the final result into a scalar result
	m_result=new CScalarResult<float64_t>(result);
	SG_REF(m_result);
}

}
#endif // HAVE_EIGEN3
