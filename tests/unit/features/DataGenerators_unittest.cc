/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Thoralf Klein, Bjoern Esser
 */

#include <gtest/gtest.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/features/streaming/generators/GaussianBlobsDataGenerator.h>
#include <shogun/features/streaming/generators/MeanShiftDataGenerator.h>

using namespace shogun;

TEST(GaussianBlobsDataGenerator,get_next_example1)
{
	index_t num_blobs=1;
	float64_t distance=3;
	float64_t epsilon=2;
	float64_t angle=Math::PI/4;
	index_t num_samples=50000;

	auto gen=std::make_shared<GaussianBlobsDataGenerator>(num_blobs,
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

	SGVector<float64_t> mean=Statistics::matrix_mean(samples, false);
	SGMatrix<float64_t> cov=Statistics::covariance_matrix(samples);

    /* rougly ensures right results, if test fails, set a bit larger */
    float64_t accuracy=2e-1;

	/* matrix is expected to look like [1.5, 0.5; 0.5, 1.5] */
	EXPECT_NEAR(cov(0,0), 1.5, accuracy);
	EXPECT_NEAR(cov(0,1), 0.5, accuracy);
	EXPECT_NEAR(cov(1,0), 0.5, accuracy);
	EXPECT_NEAR(cov(1,1), 1.5, accuracy);

	/* mean is supposed to do [0, 0] */
	EXPECT_LE(Math::abs(mean[0]-0), accuracy);
	EXPECT_LE(Math::abs(mean[1]-0), accuracy);


}

TEST(GaussianBlobsDataGenerator,get_next_example2)
{
	index_t num_blobs=3;
	float64_t distance=3;
	float64_t epsilon=2;
	float64_t angle=Math::PI/4;
	index_t num_samples=50000;

	auto gen=std::make_shared<GaussianBlobsDataGenerator>(num_blobs,
			distance, epsilon, angle);

	/* and another one */
	SGMatrix<float64_t> samples2(2, num_samples);
	gen->set_blobs_model(num_blobs, distance, epsilon, angle);

	for (index_t i=0; i<num_samples; ++i)
	{
		gen->get_next_example();
		SGVector<float64_t> sample=gen->get_vector();
		samples2(0,i)=sample[0];
		samples2(1,i)=sample[1];
		gen->release_example();
	}

	SGVector<float64_t> mean2=Statistics::matrix_mean(samples2, false);
	SGMatrix<float64_t> cov2=Statistics::covariance_matrix(samples2);

    /* rougly ensures right results, if test fails, set a bit larger */
    float64_t accuracy=2e-1;

	/* matrix is expected to look like [7.55, 0.55; 0.55, 7.55] */
	EXPECT_NEAR(cov2(0,0), 7.55, accuracy);
	EXPECT_NEAR(cov2(0,1), 0.55, accuracy);
	EXPECT_NEAR(cov2(1,0), 0.55, accuracy);
	EXPECT_NEAR(cov2(1,1), 7.55, accuracy);

	/* mean is supposed to do [3, 3] */
	EXPECT_LE(Math::abs(mean2[0]-3), accuracy);
	EXPECT_LE(Math::abs(mean2[1]-3), accuracy);


}

TEST(MeanShiftDataGenerator,get_next_example)
{
	index_t dimension=3;
	index_t mean_shift=100;
	index_t num_runs=1000;

	auto gen=std::make_shared<MeanShiftDataGenerator>(mean_shift,
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
	auto features=
			gen->get_streamed_features(num_runs)->as<DenseFeatures<float64_t>>();
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




}
