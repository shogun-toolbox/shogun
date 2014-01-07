/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012-2013 Heiko Strathmann
 */

#include <base/init.h>
#include <mathematics/Statistics.h>
#include <features/streaming/generators/GaussianBlobsDataGenerator.h>
#include <features/streaming/generators/MeanShiftDataGenerator.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(GaussianBlobsDataGenerator,get_next_example)
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

	SGVector<float64_t> mean=CStatistics::matrix_mean(samples, false);
	SGMatrix<float64_t>::transpose_matrix(samples.matrix, samples.num_rows,
			samples.num_cols);
#ifdef HAVE_LAPACK
	SGMatrix<float64_t> cov=CStatistics::covariance_matrix(samples);
#endif // HAVE_LAPACK
    //mean.display_vector("mean");
	//cov.display_matrix("cov");

    /* rougly ensures right results, set to 0.3 for now, if test fails, set a
     * bit larger */
    float64_t accuracy=0.5;

	/* matrix is expected to look like [1.5, 0.5; 0.5, 1.5] */
#ifdef HAVE_LAPACK
	EXPECT_LE(CMath::abs(cov(0,0)-1.5), accuracy);
	EXPECT_LE(CMath::abs(cov(0,1)-0.5), accuracy);
	EXPECT_LE(CMath::abs(cov(1,0)-0.5), accuracy);
	EXPECT_LE(CMath::abs(cov(1,1)-1.5), accuracy);
#endif // HAVE_LAPACK

	/* mean is supposed to do [0, 0] */
	EXPECT_LE(CMath::abs(mean[0]-0), 0.1);
	EXPECT_LE(CMath::abs(mean[1]-0), 0.1);

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

	SGVector<float64_t> mean2=CStatistics::matrix_mean(samples2, false);
	SGMatrix<float64_t>::transpose_matrix(samples2.matrix, samples2.num_rows,
			samples2.num_cols);
#ifdef HAVE_LAPACK
	SGMatrix<float64_t> cov2=CStatistics::covariance_matrix(samples2);
#endif // HAVE_LAPACK
    //mean2.display_vector("mean2");
	//cov2.display_matrix("cov2");


	/* matrix is expected to look like [7.55, 0.55; 0.55, 7.55] */
#ifdef HAVE_LAPACK
	EXPECT_LE(CMath::abs(cov2(0,0)-7.55), accuracy);
	EXPECT_LE(CMath::abs(cov2(0,1)-0.55), accuracy);
	EXPECT_LE(CMath::abs(cov2(1,0)-0.55), accuracy);
	EXPECT_LE(CMath::abs(cov2(1,1)-7.55), accuracy);
#endif // HAVE_LAPACK

	/* mean is supposed to do [3, 3] */
	EXPECT_LE(CMath::abs(mean2[0]-3), accuracy);
	EXPECT_LE(CMath::abs(mean2[1]-3), accuracy);

	SG_UNREF(gen);
}

TEST(MeanShiftDataGenerator,get_next_example)
{
	index_t dimension=3;
	index_t mean_shift=100;
	index_t num_runs=1000;

	CMeanShiftDataGenerator* gen=new CMeanShiftDataGenerator(mean_shift,
			dimension, 0);

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
	//avg.display_vector("mean_shift");

	/* roughly assert correct model parameters */
	EXPECT_LE(avg[0]-mean_shift, mean_shift/100);
	for (index_t i=1; i<dimension; ++i)
	{
		EXPECT_LE(avg[i], 0.5);
		EXPECT_GE(avg[i], -0.5);
	}

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
	//avg.display_vector("mean_shift");

	ASSERT(avg[0]-mean_shift<mean_shift/100);
	for (index_t i=1; i<dimension; ++i)
		ASSERT(avg[i]<0.5 && avg[i]>-0.5);

	SG_UNREF(features);

	SG_UNREF(gen);
}
