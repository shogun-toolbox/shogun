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
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/computation/jobresult/ScalarResult.h>
#include <shogun/lib/computation/jobresult/VectorResult.h>
#include <shogun/mathematics/linalg/linop/DenseMatrixOperator.h>
#include <shogun/mathematics/linalg/ratapprox/logdet/computation/aggregator/IndividualJobResultAggregator.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(IndividualJobResultAggregator, finalize)
{
	const index_t size=2;
	SGVector<complex128_t> result_vector(size);
	result_vector.set_const(complex128_t(0.0, 1.0));

	SGMatrix<float64_t> m(size, size);
	m(0,0)=1.0;
	m(0,1)=0.0;
	m(1,0)=0.0;
	m(1,1)=1.0;
	CDenseMatrixOperator<float64_t>* op=new CDenseMatrixOperator<float64_t>(m);
	SG_REF(op);

	CVectorResult<complex128_t> *job_result
		=new CVectorResult<complex128_t>(result_vector);
	SG_REF(job_result);

	SGVector<float64_t> sample(size);
	sample.set_const(1.0);

	const float64_t const_multiplier=1.0;

	CIndividualJobResultAggregator* agg
		=new CIndividualJobResultAggregator(op, sample, const_multiplier);
	SG_REF(agg);

	agg->submit_result(job_result);
	agg->finalize();

	CScalarResult<float64_t>* final_result
		=dynamic_cast<CScalarResult<float64_t>*>(agg->get_final_result());
	float64_t result=final_result->get_result();

	EXPECT_NEAR(result, 2.0, 1E-15);

	SG_UNREF(job_result);
	SG_UNREF(agg);
	SG_UNREF(op);
}
#endif // HAVE_EIGEN3
