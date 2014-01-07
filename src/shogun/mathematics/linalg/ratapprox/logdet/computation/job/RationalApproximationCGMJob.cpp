/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Soumyajit De
 */

#include <lib/config.h>

#ifdef HAVE_EIGEN3
#include <lib/SGVector.h>
#include <lib/computation/jobresult/ScalarResult.h>
#include <mathematics/eigen3.h>
#include <mathematics/linalg/linop/LinearOperator.h>
#include <mathematics/linalg/linsolver/CGMShiftedFamilySolver.h>
#include <mathematics/linalg/ratapprox/logdet/computation/job/RationalApproximationCGMJob.h>
#include <base/Parameter.h>

using namespace Eigen;

namespace shogun
{

CRationalApproximationCGMJob::CRationalApproximationCGMJob()
	: CIndependentJob()
{
	init();
}

CRationalApproximationCGMJob::CRationalApproximationCGMJob(
	CStoreScalarAggregator<float64_t>* aggregator,
	CCGMShiftedFamilySolver* linear_solver,
	CLinearOperator<float64_t>* linear_operator,
	SGVector<float64_t> vector,
	SGVector<complex128_t> shifts,
	SGVector<complex128_t> weights,
	float64_t const_multiplier)
	: CIndependentJob((CJobResultAggregator*)aggregator)
{
	init();

	m_linear_solver=linear_solver;
	SG_REF(m_linear_solver);

	m_operator=linear_operator;
	SG_REF(m_operator);

	m_vector=vector;

	m_shifts=shifts;
	m_weights=weights;
	m_const_multiplier=const_multiplier;
}

void CRationalApproximationCGMJob::init()
{
	m_linear_solver=NULL;
	m_operator=NULL;
	m_const_multiplier=0.0;

	SG_ADD((CSGObject**)&m_linear_solver, "linear_solver",
		"Linear solver for complex-shifted system", MS_NOT_AVAILABLE);

	SG_ADD((CSGObject**)&m_operator, "linear_operator",
		"Linear operator", MS_NOT_AVAILABLE);

	SG_ADD(&m_vector, "trace_sample",
		"Sample vector to apply linear operator on", MS_NOT_AVAILABLE);

	SG_ADD(&m_weights, "complex_shifts",
		"Shifts in the linear systems to be solved", MS_NOT_AVAILABLE);

	SG_ADD(&m_weights, "complex_weights",
		"Weights to be multiplied to the solution vector", MS_NOT_AVAILABLE);

	SG_ADD(&m_const_multiplier, "constant_multiplier",
		"Constant multiplier to be multiplied with the final solution", MS_NOT_AVAILABLE);
}

CRationalApproximationCGMJob::~CRationalApproximationCGMJob()
{
	SG_UNREF(m_linear_solver);
	SG_UNREF(m_operator);
}

void CRationalApproximationCGMJob::compute()
{
	SG_DEBUG("Entering\n");

	REQUIRE(m_aggregator, "Job result aggregator is not set!\n");
	REQUIRE(m_operator, "Operator is not set!\n");
	REQUIRE(m_vector.vector, "Vector is not set!\n");
	REQUIRE(m_shifts.vector, "Shifts are not set!\n");
	REQUIRE(m_weights.vector, "Weights are not set!\n");
	REQUIRE(m_operator->get_dimension()==m_vector.vlen,
		"Dimension mismatch! %d vs %d\n", m_operator->get_dimension(), m_vector.vlen);
	REQUIRE(m_shifts.vlen==m_weights.vlen,
		"Number of shifts and weights are not equal!\n");

	// solve the linear system with the sample vector
	SGVector<complex128_t> vec=m_linear_solver->solve_shifted_weighted(
		m_operator, m_vector, m_shifts, m_weights);

	// Take negative (see CRationalApproximation for the formula)
	Map<VectorXcd> v(vec.vector, vec.vlen);
	v=-v;

	// take out the imaginary part of the result before
	// applying linear operator
	SGVector<float64_t> agg=m_operator->apply(vec.get_imag());

	// perform dot product
	Map<VectorXd> map_agg(agg.vector, agg.vlen);
	Map<VectorXd> map_vector(m_vector.vector, m_vector.vlen);
	float64_t result=map_vector.dot(map_agg);

	result*=m_const_multiplier;

	// form the final result into a scalar result and submit to the aggregator
	CScalarResult<float64_t>* final_result=new CScalarResult<float64_t>(result);
	SG_REF(final_result);

	m_aggregator->submit_result(final_result);

	SG_UNREF(final_result);

	SG_DEBUG("Leaving\n");
}

}
#endif // HAVE_EIGEN3
