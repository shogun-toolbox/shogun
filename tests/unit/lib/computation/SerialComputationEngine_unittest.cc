/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Soumyajit De
 */

#include <lib/common.h>

#ifdef HAVE_EIGEN3
#include <mathematics/eigen3.h>

#if EIGEN_VERSION_AT_LEAST(3,1,0)
#include <unsupported/Eigen/MatrixFunctions>
#include <lib/SGVector.h>
#include <lib/SGMatrix.h>
#include <mathematics/linalg/linop/DenseMatrixOperator.h>
#include <lib/computation/jobresult/ScalarResult.h>
#include <lib/computation/aggregator/StoreScalarAggregator.h>
#include <mathematics/linalg/ratapprox/logdet/computation/job/DenseExactLogJob.h>
#include <lib/computation/engine/SerialComputationEngine.h>
#include <mathematics/Statistics.h>
#include <gtest/gtest.h>

using namespace Eigen;
using namespace shogun;

TEST(SerialComputationEngine, dense_log_det)
{
	CSerialComputationEngine e;
	const index_t size=2;

	// create the matrix whose log-det has to be found
	SGMatrix<float64_t> mat(size, size);
	SGMatrix<float64_t> log_mat(size, size);
	mat(0,0)=2.0;
	mat(0,1)=1.0;
	mat(1,0)=1.0;
	mat(1,1)=3.0;
	Map<MatrixXd> m(mat.matrix, mat.num_rows, mat.num_cols);
	Map<MatrixXd> log_m(log_mat.matrix, log_mat.num_rows, log_mat.num_cols);
	log_m=m.log();

	// create linear operator and aggregator
	CDenseMatrixOperator<float64_t>* log_op=new CDenseMatrixOperator<float64_t>(log_mat);
	SG_REF(log_op);
	CStoreScalarAggregator<float64_t>* agg=new CStoreScalarAggregator<float64_t>;
	SG_REF(agg);

	// create jobs with unit-vectors to extract the trace of log(mat)
	for (index_t i=0; i<size; ++i)
	{
		SGVector<float64_t> s(size);
		s.set_const(0.0);
		s[i]=1.0;
		CDenseExactLogJob *job=new CDenseExactLogJob((CJobResultAggregator*)agg,
			log_op, s);
		SG_REF(job);
		// submit the job to the computation engine
		e.submit_job(job);
		SG_UNREF(job);
	}

	// wait for all the jobs to be computed in the computation engine
	e.wait_for_all();
	// its really important we call finalize before getting the final result
	agg->finalize();

	CScalarResult<float64_t>* r=dynamic_cast<CScalarResult<float64_t>*>
		(agg->get_final_result());

	EXPECT_NEAR(r->get_result(), CStatistics::log_det(mat), 1E-15);

	SG_UNREF(log_op);
	SG_UNREF(agg);
}
#endif // EIGEN_VERSION_AT_LEAST(3,1,0)
#endif // HAVE_EIGEN3
