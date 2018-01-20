/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Soumyajit De
 */
#include <gtest/gtest.h>

#include <shogun/lib/common.h>

#include <shogun/mathematics/eigen3.h>

#include <omp.h>
#include <shogun/lib/DynamicArray.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/mathematics/linalg/linop/DenseMatrixOperator.h>
#include <shogun/mathematics/linalg/ratapprox/logdet/opfunc/DenseMatrixExactLog.h>
#include <unsupported/Eigen/MatrixFunctions>

using namespace shogun;
using namespace Eigen;

TEST(DenseMatrixExactLog, dense_log_det)
{
	const index_t size=2;
	SGMatrix<float64_t> mat(size, size);
	mat(0,0)=2.0;
	mat(0,1)=1.0;
	mat(1,0)=1.0;
	mat(1,1)=3.0;
	CDenseMatrixOperator<float64_t>* op=new CDenseMatrixOperator<float64_t>(mat);
	SG_REF(op);

	// create operator function with the operator and the engine to submit jobs
	CDenseMatrixExactLog* op_func = new CDenseMatrixExactLog(op);
	SG_REF(op_func);

	// its really important we call the precompute on the operator function
	op_func->precompute();

	// for storing the result solve returns
	CDynamicArray<float64_t> aggregators;

	// create samples for extracting the trace of log(C) and submit

	for (index_t i=0; i<size; ++i)
	{
		SGVector<float64_t> s(size);
		s.set_const(0.0);
		s[i]=1.0;

		float64_t agg = op_func->solve(s);
		aggregators.append_element(agg);
	}

	// use the aggregators to find the final result
	int32_t num_aggregates=aggregators.get_num_elements();
	float64_t result=0.0;

	for (int32_t i=0; i<num_aggregates; ++i)
	{
		result += aggregators.get_element(i);
	}


	EXPECT_NEAR(result, CStatistics::log_det(mat), 1E-15);

	// clean up
	SG_UNREF(op_func);
	SG_UNREF(op);
}
