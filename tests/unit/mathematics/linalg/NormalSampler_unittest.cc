/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Soumyajit De
 */

#include <shogun/lib/common.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/linalg/ratapprox/tracesampler/NormalSampler.h>
#include <gtest/gtest.h>

using namespace shogun;
using namespace Eigen;

TEST(NormalSampler, sample)
{
	const index_t dimension=2;
	const index_t num_samples=5000;
	SGMatrix<float64_t> samples(num_samples, dimension);

	CNormalSampler sampler(dimension);
	sampler.precompute();
	for (index_t i=0; i<num_samples; ++i)
	{
		SGVector<float64_t> s=sampler.sample(0);
		for (index_t j=0; j<dimension; ++j)
			samples(i,j)=s[j];
	}

	SGVector<float64_t> mean=CStatistics::matrix_mean(samples);
	Map<VectorXd> map_mean(mean.vector, mean.vlen);
	EXPECT_NEAR((map_mean-VectorXd::Zero(dimension)).norm(), 0.0, 0.1);
	SGMatrix<float64_t>::transpose_matrix(samples.matrix, samples.num_rows, samples.num_cols); // TODO: refactor sample_from_gaussian to return column vectors!
	SGMatrix<float64_t> cov=CStatistics::covariance_matrix(samples);
	Map<MatrixXd> map_cov(cov.matrix, cov.num_rows, cov.num_cols);
	EXPECT_NEAR((map_cov-MatrixXd::Identity(dimension, dimension)).norm(),
		0.0, 0.1);
}


