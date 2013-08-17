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
#include <shogun/mathematics/logdet/DenseMatrixOperator.h>
#include <shogun/lib/computation/job/ScalarResult.h>
#include <shogun/lib/computation/job/StoreScalarAggregator.h>
#include <shogun/mathematics/logdet/CGMShiftedFamilySolver.h>
#include <shogun/lib/computation/job/RationalApproximationCGMJob.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(RationalApproximationCGMJob, compute)
{
	const int32_t size=4;
	SGMatrix<float64_t> m(size, size);
	m.set_const(0.0);

	// diagonal Hermintian matrix
	for (index_t i=0; i<size; ++i)
		m(i,i)=i+1;

	CDenseMatrixOperator<float64_t>* linear_operator
		=new CDenseMatrixOperator<float64_t>(m);
	SG_REF(linear_operator);

	SGVector<float64_t> sample(size);
	sample.set_const(0.5);

	SGVector<complex64_t> shifts(1);
	shifts[0]=complex64_t(0.0, 0.01);

	SGVector<complex64_t> weights(1);
	weights[0]=1.0;

	float const_multiplier=1.0;

	CCGMShiftedFamilySolver* linear_solver=new CCGMShiftedFamilySolver();
	SG_REF(linear_solver);
	CStoreScalarAggregator<float64_t>* aggregator
		=new CStoreScalarAggregator<float64_t>();
	SG_REF(aggregator);

	CRationalApproximationCGMJob* job=new CRationalApproximationCGMJob(
		aggregator, linear_solver, linear_operator, sample, shifts, weights,
		const_multiplier);
	SG_REF(job);

	job->compute();
	aggregator->finalize();
	CScalarResult<float64_t>* final_result
		=dynamic_cast<CScalarResult<float64_t>*>(aggregator->get_final_result());
	float64_t result=final_result->get_result();

	EXPECT_NEAR(result, 0.02929525851272159198, 1E-15);

	SG_UNREF(job);
	SG_UNREF(aggregator);
	SG_UNREF(linear_operator);
	SG_UNREF(linear_solver);
}
#endif //HAVE_EIGEN3
