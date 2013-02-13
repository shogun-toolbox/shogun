/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 *
 * This file demonstrates how to use data generators based on the streaming
 * features framework
 */

#include <shogun/base/init.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/features/streaming/generators/MeanShiftDataGenerator.h>
#include <shogun/features/streaming/generators/GaussianBlobsDataGenerator.h>

using namespace shogun;

void test_mean_shift()
{
	index_t dimension=3;
	index_t mean_shift=100;
	index_t num_runs=1000;

	CMeanShiftDataGenerator* gen=new CMeanShiftDataGenerator(mean_shift, dimension);

	SGVector<float64_t> avg(dimension);
	avg.zero();

	for (index_t i=0; i<num_runs; ++i)
	{
		gen->get_next_example();
		avg.add(gen->get_vector());
		gen->release_example();
	}

	/* average */
	avg.scale(1.0/num_runs);
	avg.display_vector("mean_shift");

	/* roughly assert correct model parameters */
	ASSERT(avg[0]-mean_shift<mean_shift/100);
	for (index_t i=1; i<dimension; ++i)
		ASSERT(avg[i]<0.5 && avg[i]>-0.5);

	/* draw whole matrix and test that too */
	CDenseFeatures<float64_t>* features=
			(CDenseFeatures<float64_t>*)gen->get_streamed_features(num_runs);
	avg=SGVector<float64_t>(dimension);

	for (index_t i=0; i<dimension; ++i)
	{
		float64_t sum=0;
		for (index_t j=0; j<num_runs; ++j)
			sum+=features->get_feature_matrix()(i, j);

		avg[i]=sum/num_runs;
	}
	avg.display_vector("mean_shift");

	ASSERT(avg[0]-mean_shift<mean_shift/100);
	for (index_t i=1; i<dimension; ++i)
		ASSERT(avg[i]<0.5 && avg[i]>-0.5);

	SG_UNREF(features);

	SG_UNREF(gen);
}

void gaussian_blobs()
{
	index_t num_blobs=1;
	float64_t distance=3;
	float64_t epsilon=2;
	float64_t angle=CMath::PI/4;
	index_t num_samples=10000;

	CGaussianBlobsDataGenerator* gen=new CGaussianBlobsDataGenerator(num_blobs,
			distance, epsilon, angle);

	/* two dimensional samples */
	SGMatrix<float64_t> samples(2, num_samples);

	for (index_t i=0; i<num_samples; ++i)
	{
		gen->get_next_example();
		SGVector<float64_t> sample=gen->get_vector();
		samples(0,i)=sample[0];
		samples(1,i)=sample[1];
		gen->release_example();
	}

	CStatistics::matrix_mean(samples, false).display_vector("mean");
	SGMatrix<float64_t>::transpose_matrix(samples.matrix, samples.num_rows,
			samples.num_cols);
	CStatistics::covariance_matrix(samples).display_matrix("covariance");

	/* matrix is expected to look like [1.5, 0.5; 0.5, 1.5]
	 * mean is supposed to do [0, 0] */

	/* and another one */
	SGMatrix<float64_t> samples2(2, num_samples);
	num_blobs=3;
	gen->set_blobs_model(num_blobs, distance, epsilon, angle);

	for (index_t i=0; i<num_samples; ++i)
	{
		gen->get_next_example();
		SGVector<float64_t> sample=gen->get_vector();
		samples2(0,i)=sample[0];
		samples2(1,i)=sample[1];
		gen->release_example();
	}

	CStatistics::matrix_mean(samples2, false).display_vector("mean2");
	SGMatrix<float64_t>::transpose_matrix(samples2.matrix, samples2.num_rows,
			samples2.num_cols);
	CStatistics::covariance_matrix(samples2).display_matrix("covariance2");

	/* matrix is expected to look like [7.55, 0.55; 7.55, 0.55]
	 * mean is supposed to do [3, 3] */

	SG_UNREF(gen);
}

int main(int argc, char** argv)
{
	init_shogun_with_defaults();

	test_mean_shift();
	gaussian_blobs();

	exit_shogun();
	return 0;
}

