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
#include <shogun/mathematics/eigen3.h>

#if EIGEN_VERSION_AT_LEAST(3,1,0)
#include <unsupported/Eigen/MatrixFunctions>

#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/DynamicObjectArray.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/mathematics/logdet/DenseMatrixOperator.h>
#include <shogun/mathematics/logdet/DenseMatrixExactLog.h>
#include <shogun/lib/computation/job/ScalarResult.h>
#include <shogun/lib/computation/job/StoreScalarAggregator.h>
#include <shogun/lib/computation/job/DenseExactLogJob.h>
#include <shogun/lib/computation/engine/SerialComputationEngine.h>
#include <gtest/gtest.h>

using namespace shogun;
using namespace Eigen;

TEST(DenseMatrixExactLog, dense_log_det)
{
	// the computation engine for this
	CSerialComputationEngine* e=new CSerialComputationEngine;
	SG_REF(e);

	// create the linear operator whose log-det has to be computed	
	const index_t size=2;
	SGMatrix<float64_t> mat(size, size);
	mat(0,0)=2.0;
	mat(0,1)=1.0;
	mat(1,0)=1.0;
	mat(1,1)=3.0;
	CDenseMatrixOperator<float64_t>* op=new CDenseMatrixOperator<float64_t>(mat);
	SG_REF(op);
	
	// create operator function with the operator and the engine to submit jobs
	CDenseMatrixExactLog *op_func=new CDenseMatrixExactLog(op, e);
	SG_REF(op_func);

	// its really important we call the precompute on the operato function
	op_func->precompute();

	// for storing the aggregators that submit_jobs return
	CDynamicObjectArray aggregators;

	// create samples for extracting the trace of log(C) and submit
	for (index_t i=0; i<size; ++i)
	{
		SGVector<float64_t> s(size);
		s.set_const(0.0);
		s[i]=1.0;

		CJobResultAggregator* agg=op_func->submit_jobs(s);
		aggregators.append_element(agg);
		SG_UNREF(agg);
	}

	// wait for all jobs to be computed
	e->wait_for_all();

	// use the aggregators to find the final result
	int32_t num_aggregates=aggregators.get_num_elements();
	float64_t result=0.0;

	for (int32_t i=0; i<num_aggregates; ++i)
	{
		CJobResultAggregator* agg=dynamic_cast<CJobResultAggregator*>
			(aggregators.get_element(i));
		agg->finalize();
		CScalarResult<float64_t>* r=dynamic_cast<CScalarResult<float64_t>*>
			(agg->get_final_result());
		// its important that we don't just unref the result here
		result+=r->get_result();
		SG_UNREF(agg);
	}
	
	// clear all aggregators
	aggregators.clear_array();

	EXPECT_NEAR(result, CStatistics::log_det(mat), 1E-15);

	// clean up
	SG_UNREF(op_func);
	SG_UNREF(op);
	SG_UNREF(e);
}
#endif // EIGEN_VERSION_AT_LEAST(3,1,0)
#endif // HAVE_EIGEN3
