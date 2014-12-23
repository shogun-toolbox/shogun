/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Parijat Mazumdar
 */

#include <shogun/labels/MulticlassLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/clustering/KMeans.h>
#include <shogun/clustering/KMeansMiniBatch.h>
#include <shogun/distance/EuclideanDistance.h>
#include <gtest/gtest.h>

using namespace shogun;

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
	CEuclideanDistance* distance=new CEuclideanDistance(features, features);
	CKMeans* clustering=new CKMeans(2, distance,initial_centers);

	for (int32_t loop=0; loop<10; loop++)
	{
		clustering->train(features);

		CMulticlassLabels* result=CLabelsFactory::to_multiclass(clustering->apply());

		EXPECT_EQ(0.000000, result->get_label(0));
		EXPECT_EQ(0.000000, result->get_label(1));
		EXPECT_EQ(1.000000, result->get_label(2));
		EXPECT_EQ(1.000000, result->get_label(3));
	
		CDenseFeatures<float64_t>* learnt_centers=CDenseFeatures<float64_t>::obtain_from_generic(distance->get_lhs());
		SGMatrix<float64_t> learnt_centers_matrix=learnt_centers->get_feature_matrix();
	
		EXPECT_EQ(0, learnt_centers_matrix(0,0));
		EXPECT_EQ(2, learnt_centers_matrix(0,1));
		EXPECT_EQ(5, learnt_centers_matrix(1,0));
		EXPECT_EQ(5, learnt_centers_matrix(1,1));
		
		SG_UNREF(learnt_centers);
		SG_UNREF(result);
	}

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
		CDenseFeatures<float64_t>* learnt_centers=CDenseFeatures<float64_t>::obtain_from_generic(distance->get_lhs());
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

	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(rect);
	CEuclideanDistance* distance=new CEuclideanDistance(features, features);
	CKMeansMiniBatch* clustering=new CKMeansMiniBatch(1, distance,initial_centers);

	for (int32_t loop=0; loop<10; loop++)
	{
		clustering->set_mbKMeans_params(4,1000);
		clustering->train(features);
		CDenseFeatures<float64_t>* learnt_centers=CDenseFeatures<float64_t>::obtain_from_generic(distance->get_lhs());
		SGMatrix<float64_t> learnt_centers_matrix=learnt_centers->get_feature_matrix();

		EXPECT_NEAR(1, learnt_centers_matrix(0,0), 0.0001);
		EXPECT_NEAR(500,learnt_centers_matrix(1,0), 0.0001);

		SG_UNREF(learnt_centers);
	}

	SG_UNREF(clustering);
	SG_UNREF(features);
}

