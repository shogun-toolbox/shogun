/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Saurabh Mahindre, Heiko Strathmann, Viktor Gal, parijat,
 *          Bjoern Esser, Soeren Sonnenburg
 */

#include <gtest/gtest.h>
#include <shogun/clustering/KMeans.h>
#include <shogun/clustering/KMeansMiniBatch.h>
#include <shogun/distance/EuclideanDistance.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/lib/observers/ParameterObserver.h>
#include <shogun/lib/observers/ParameterObserverLogger.h>

using namespace shogun;

void check_consistency_observable(
    const CKMeans* kmeans, ParameterObserver* observer)
{
	auto total_observations = observer->get<int32_t>("num_observations");
	auto observation = observer->get_observation(total_observations - 1);
	auto centers = observation->get<SGMatrix<float64_t>>("mus");

	EXPECT_TRUE(
	    centers.equals(kmeans->get<SGMatrix<float64_t>>("cluster_centers")));
}

TEST(KMeans, manual_center_initialization_test)
{
	/*create a rectangle with four points as (0,0) (0,10) (2,0) (2,10)*/
	SGMatrix<float64_t> rect(2, 4);
	rect(0,0)=0;
	rect(0,1)=0;
	rect(0,2)=2;
	rect(0,3)=2;
	rect(1,0)=0;
	rect(1,1)=10;
	rect(1,2)=0;
	rect(1,3)=10;

	/*choose local minima points (0,5) (2,5) as initial centers*/
	SGMatrix<float64_t> initial_centers(2,2);
	initial_centers(0,0)=0;
	initial_centers(0,1)=2;
	initial_centers(1,0)=5;
	initial_centers(1,1)=5;

	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(rect);
	SG_REF(features);
	CEuclideanDistance* distance=new CEuclideanDistance(features, features);
	CKMeans* clustering=new CKMeans(2, distance,initial_centers);

	CParameterObserverLogger* observer = new CParameterObserverLogger();
	clustering->subscribe(observer);

	for (int32_t loop=0; loop<10; loop++)
	{
		clustering->train(features);

		CMulticlassLabels* result =
		    clustering->apply()->as<CMulticlassLabels>();

		EXPECT_EQ(0.000000, result->get_label(0));
		EXPECT_EQ(0.000000, result->get_label(1));
		EXPECT_EQ(1.000000, result->get_label(2));
		EXPECT_EQ(1.000000, result->get_label(3));

		CDenseFeatures<float64_t>* learnt_centers=distance->get_lhs()->as<CDenseFeatures<float64_t>>();
		SGMatrix<float64_t> learnt_centers_matrix=learnt_centers->get_feature_matrix();

		EXPECT_EQ(0, learnt_centers_matrix(0,0));
		EXPECT_EQ(2, learnt_centers_matrix(0,1));
		EXPECT_EQ(5, learnt_centers_matrix(1,0));
		EXPECT_EQ(5, learnt_centers_matrix(1,1));

		check_consistency_observable(clustering, observer);
		SG_UNREF(learnt_centers);
		SG_UNREF(result);
	}

	clustering->unsubscribe(observer);

	SG_UNREF(observer)
	SG_UNREF(clustering);
	SG_UNREF(features);
}

TEST(KMeans, KMeanspp_center_initialization_test)
{
	/*create a rectangle with four points as (0,0) (0,10) (2,0) (2,10)*/
	SGMatrix<float64_t> rect(2, 4);
	rect(0,0)=0;
	rect(0,1)=0;
	rect(0,2)=2;
	rect(0,3)=2;
	rect(1,0)=0;
	rect(1,1)=10;
	rect(1,2)=0;
	rect(1,3)=10;

	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(rect);
	SG_REF(features);
	CEuclideanDistance* distance=new CEuclideanDistance(features, features);
	CKMeans* clustering=new CKMeans(4, distance,true);

	for (int32_t loop=0; loop<10; loop++)
	{
		clustering->train(features);
		CDenseFeatures<float64_t>* learnt_centers=distance->get_lhs()->as<CDenseFeatures<float64_t>>();
		SGMatrix<float64_t> learnt_centers_matrix=learnt_centers->get_feature_matrix();
		SGVector<int32_t> count=SGVector<int32_t>(4);
		count.zero();
		for (int32_t c=0; c<4; c++)
		{
			if (learnt_centers_matrix(0,c)==0 && learnt_centers_matrix(1,c)==0)
			{
				count[0]++;
			}
			else if (learnt_centers_matrix(0,c)==0 && learnt_centers_matrix(1,c)==10)
			{
				count[1]++;
			}
			else if (learnt_centers_matrix(0,c)==2 && learnt_centers_matrix(1,c)==0)
			{
				count[2]++;
			}
			else if (learnt_centers_matrix(0,c)==2 && learnt_centers_matrix(1,c)==10)
			{
				count[3]++;
			}
		}

		EXPECT_EQ(1, count[0]);
		EXPECT_EQ(1, count[1]);
		EXPECT_EQ(1, count[2]);
		EXPECT_EQ(1, count[3]);

		SG_UNREF(learnt_centers);
	}

	SG_UNREF(clustering);
	SG_UNREF(features);
}

TEST(KMeans, minibatch_training_test)
{
	/*create a rectangle with four points as (0,0) (0,1000) (2,0) (2,1000)*/
	SGMatrix<float64_t> rect(2, 4);
	rect(0,0)=0;
	rect(0,1)=0;
	rect(0,2)=2;
	rect(0,3)=2;
	rect(1,0)=0;
	rect(1,1)=1000;
	rect(1,2)=0;
	rect(1,3)=1000;

	SGMatrix<float64_t> initial_centers(2,1);
	initial_centers(0,0)=0;
	initial_centers(1,0)=0;

	auto features = some<CDenseFeatures<float64_t>>(rect);
	auto distance = some<CEuclideanDistance>(features, features);
	auto clustering = some<CKMeansMiniBatch>(1, distance, initial_centers);

	for (int32_t loop=0; loop<10; ++loop)
	{
		clustering->put<int32_t>("max_iter", 1000);
		clustering->put<int32_t>("batch_size", 4);
		clustering->train(features);
		auto learnt_centers=wrap(distance->get_lhs()->as<CDenseFeatures<float64_t>>());
		SGMatrix<float64_t> learnt_centers_matrix=learnt_centers->get_feature_matrix();

		EXPECT_NEAR(1, learnt_centers_matrix(0,0), 0.0001);
		EXPECT_NEAR(500,learnt_centers_matrix(1,0), 0.0001);
	}
}

TEST(KMeans, fixed_centers)
{
	/*create a rectangle with four points as (0,0) (0,10) (20,0) (20,10)*/
	SGMatrix<float64_t> rect(2, 4);
	rect(0,0)=0;
	rect(1,0)=0;
	rect(0,1)=0;
	rect(1,1)=10;
	rect(0,2)=20;
	rect(1,2)=0;
	rect(0,3)=20;
	rect(1,3)=10;

	/*choose local minima points (but not exact means) (0,4) (20,4) as initial centers */
	SGMatrix<float64_t> initial_centers(2,2);
	initial_centers(0,0)=0;
	initial_centers(1,0)=4;
	initial_centers(0,1)=20;
	initial_centers(1,1)=4;

	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(rect);
	SG_REF(features);
	CEuclideanDistance* distance=new CEuclideanDistance(features, features);
	CKMeans* clustering=new CKMeans(2, distance,initial_centers);
	clustering->put<bool>("fixed_centers", true);

	clustering->train(features);
	CDenseFeatures<float64_t>* learnt_centers=(CDenseFeatures<float64_t>*)distance->get_lhs();

	ASSERT_NE(learnt_centers, (CDenseFeatures<float64_t>*)NULL);
	SGMatrix<float64_t> c=learnt_centers->get_feature_matrix();

	EXPECT_NEAR(c(0,0), 0.0, 10E-12);
	EXPECT_NEAR(c(1,0), 5.0, 10E-12);
	EXPECT_NEAR(c(0,1), 20.0, 10E-12);
	EXPECT_NEAR(c(1,1), 5.0, 10E-12);

	SG_UNREF(clustering);
	SG_UNREF(features);
	SG_UNREF(learnt_centers);
}

TEST(KMeans, random_centers_init)
{
	/* Random centers should initialize with unique centers  */
	SGMatrix<float64_t> rect(2, 3);
	rect(0,0)=0;
	rect(1,0)=0;
	rect(0,1)=10;
	rect(1,1)=10;
	rect(0,2)=20;
	rect(1,2)=20;

	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(rect);
	SG_REF(features);
	CEuclideanDistance* distance=new CEuclideanDistance(features, features);
	CKMeans* clustering=new CKMeans(3, distance);

	clustering->train(features);
	CDenseFeatures<float64_t>* learnt_centers=(CDenseFeatures<float64_t>*)distance->get_lhs();
	ASSERT_NE(learnt_centers, (CDenseFeatures<float64_t>*)NULL);
	SGMatrix<float64_t> c=learnt_centers->get_feature_matrix();

	EXPECT_NE(c(0,0),c(0,1));
	EXPECT_NE(c(0,0),c(0,2));
	EXPECT_NE(c(0,1),c(0,2));

	SG_UNREF(clustering);
	SG_UNREF(features);
	SG_UNREF(learnt_centers);
}

TEST(KMeans, random_centers_assign)
{
	/* Random centers initialization should correctly assign two very separate clusters  */
	SGMatrix<float64_t> rect(2, 4);
	rect(0,0)=0;
	rect(1,0)=0;
	rect(0,1)=0;
	rect(1,1)=10;
	rect(0,2)=20;
	rect(1,2)=0;
	rect(0,3)=20;
	rect(1,3)=10;

	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(rect);
	SG_REF(features);
	CEuclideanDistance* distance=new CEuclideanDistance(features, features);
	CKMeans* clustering=new CKMeans(2, distance);

	clustering->train(features);
	CDenseFeatures<float64_t>* learnt_centers=(CDenseFeatures<float64_t>*)distance->get_lhs();
	ASSERT_NE(learnt_centers, (CDenseFeatures<float64_t>*)NULL);
	SGMatrix<float64_t> c=learnt_centers->get_feature_matrix();

	SGVector<float64_t> count=SGVector<float64_t>(2);
	count.zero();

	if ((c(0,0)==0) && (c(1,0)==5))
		count[0]++;
	if ((c(0,1)==0) && (c(1,1)==5))
		count[0]++;
	if ((c(0,0)==20) && (c(1,0)==5))
		count[1]++;
	if ((c(0,1)==20) && (c(1,1)==5))
		count[1]++;

	if (count[0] == 0)
		EXPECT_EQ(count[1], 0);
	if (count[0] == 1)
		EXPECT_EQ(count[1], 1);

	SG_UNREF(clustering);
	SG_UNREF(features);
	SG_UNREF(learnt_centers);
}

