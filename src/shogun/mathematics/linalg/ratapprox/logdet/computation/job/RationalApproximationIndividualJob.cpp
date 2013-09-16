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
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/computation/jobresult/VectorResult.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/linalg/linop/LinearOperator.h>
#include <shogun/mathematics/linalg/linsolver/LinearSolver.h>
#include <shogun/mathematics/linalg/ratapprox/logdet/computation/job/RationalApproximationIndividualJob.h>
#include <shogun/base/Parameter.h>

using namespace Eigen;

namespace shogun
{

CRationalApproximationIndividualJob::CRationalApproximationIndividualJob()
	: CIndependentJob()
{
	init();

	SG_GCDEBUG("%s created (%p)\n", this->get_name(), this)
}

CRationalApproximationIndividualJob::CRationalApproximationIndividualJob(
	CJobResultAggregator* aggregator,
	CLinearSolver<complex64_t, float64_t>* linear_solver,
	CLinearOperator<complex64_t>* linear_operator,
	SGVector<float64_t> vector,
	complex64_t weight)
	: CIndependentJob(aggregator)
{
	init();

	m_linear_solver=linear_solver;
	SG_REF(m_linear_solver);

	m_operator=linear_operator;
	SG_REF(m_operator);

	m_vector=vector;

	m_weight=weight;

	SG_GCDEBUG("%s created (%p)\n", this->get_name(), this)
}

void CRationalApproximationIndividualJob::init()
{
	m_linear_solver=NULL;
	m_operator=NULL;
	m_weight=complex64_t(0.0);

	SG_ADD((CSGObject**)&m_linear_solver, "linear_solver",
		"Linear solver for complex system", MS_NOT_AVAILABLE);

	SG_ADD((CSGObject**)&m_operator, "shifted_operator",
		"Shifted linear operator", MS_NOT_AVAILABLE);

	SG_ADD(&m_vector, "trace_sample",
		"Sample vector to apply linear operator on", MS_NOT_AVAILABLE);

	SG_ADD(&m_weight, "complex_weight",
		"Weight to be multiplied to the solution vector", MS_NOT_AVAILABLE);
}

CRationalApproximationIndividualJob::~CRationalApproximationIndividualJob()
{
	SG_UNREF(m_linear_solver);
	SG_UNREF(m_operator);

	SG_GCDEBUG("%s destroyed (%p)\n", this->get_name(), this)
}

void CRationalApproximationIndividualJob::compute()
{
	REQUIRE(m_aggregator, "Job result aggregator is not set!\n");
	REQUIRE(m_operator, "Operator is not set!\n");
	REQUIRE(m_vector.vector, "Vector is not set!\n");

	// solve the linear system with the sample vector
	SGVector<complex64_t> vec=m_linear_solver->solve(m_operator, m_vector);

	// multiply with the weight using Eigen3 and take negative
	// (see CRationalApproximation for the formula)
	Map<VectorXcd> v(vec.vector, vec.vlen);
	v*=m_weight;
	v=-v;

	// set as a vector result and submit to the aggregator
	CVectorResult<complex64_t>* result=new CVectorResult<complex64_t>(vec);
	SG_REF(result);

	m_aggregator->submit_result(result);

	SG_UNREF(result);
}

}
#endif // HAVE_EIGEN3
