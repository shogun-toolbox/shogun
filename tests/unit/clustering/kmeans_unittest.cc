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
#include <shogun/distance/EuclideanDistance.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(KMeans, manual_center_initialization_test)
{
	/*create a rectangle with four points as (0,0) (0,10) (2,0) (2,10)*/
	SGMatrix<float64_t> rect(2, 4);
	rect(0,0) = 0;
	rect(0,1) = 0;
	rect(0,2) = 2;
	rect(0,3) = 2;
	rect(1,0) = 0;
	rect(1,1) = 10;
	rect(1,2) = 0;
	rect(1,3) = 10;
	
	/*choose local minima points (0,5) (2,5) as initial centers*/
	float64_t mat[4] = {0,5,2,5};	
	SGMatrix<float64_t> initial_centers(mat,2,2,false);

	CDenseFeatures<float64_t>* features = new CDenseFeatures<float64_t>(rect);
	CEuclideanDistance* distance = new CEuclideanDistance(features, features);
	CKMeans* clustering=new CKMeans(2, distance,initial_centers);

	for (int32_t loop = 0; loop < 10; loop++ )
	{
		clustering->train(features);

		CMulticlassLabels* result=CLabelsFactory::to_multiclass(clustering->apply());

		EXPECT_EQ(0.000000, result->get_label(0));
		EXPECT_EQ(0.000000, result->get_label(1));
		EXPECT_EQ(1.000000, result->get_label(2));
		EXPECT_EQ(1.000000, result->get_label(3));
	
		CDenseFeatures<float64_t>* learnt_centers= (CDenseFeatures<float64_t>*) distance->get_lhs();

		SGMatrix<float64_t> learnt_centers_matrix= learnt_centers->get_feature_matrix();
	
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

