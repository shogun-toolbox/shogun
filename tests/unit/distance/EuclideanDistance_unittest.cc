/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Fernando J. Iglesias Garcia
 */

#include <shogun/distance/EuclideanDistance.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/lib/SGMatrix.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(EuclideanDistance,distance)
{
	// create two features objects with two features vectors each,
	// each column represents a feature vector
	SGMatrix<float64_t> feat_mat_lhs(2,2);
	SGMatrix<float64_t> feat_mat_rhs(2,2);

	// 1st feature vector
	feat_mat_lhs(0,0)=0;
	feat_mat_lhs(1,0)=0;
	// 2nd feature vector
	feat_mat_lhs(0,1)=0;
	feat_mat_lhs(1,1)=-1;
	// 3rd feature vector
	feat_mat_rhs(0,0)=1;
	feat_mat_rhs(1,0)=1;
	// 4th feature vector
	feat_mat_rhs(0,1)=-1;
	feat_mat_rhs(1,1)=1;

	// wrap feat_mat in Shogun features
	CDenseFeatures<float64_t>* features_lhs=new CDenseFeatures<float64_t>(feat_mat_lhs);
	CDenseFeatures<float64_t>* features_rhs=new CDenseFeatures<float64_t>(feat_mat_rhs);

	// put features into distance object to compute squared Euclidean distances
	CEuclideanDistance* euclidean=new CEuclideanDistance(features_lhs,features_rhs);
	euclidean->set_disable_sqrt(true);

	// check distances computed one by one
	EXPECT_EQ(euclidean->distance(0,0), 2);
	EXPECT_EQ(euclidean->distance(0,1), 2);
	EXPECT_EQ(euclidean->distance(1,0), 5);
	EXPECT_EQ(euclidean->distance(1,1), 5);

	// release memory
	SG_UNREF(euclidean); // the features are unref-ed here as well
}

TEST(EuclideanDistance,get_distance_matrix)
{
	// create two features objects with two features vectors each,
	// each column represents a feature vector
	SGMatrix<float64_t> feat_mat_lhs(2,2);
	SGMatrix<float64_t> feat_mat_rhs(2,2);

	// 1st feature vector
	feat_mat_lhs(0,0)=0;
	feat_mat_lhs(1,0)=0;
	// 2nd feature vector
	feat_mat_lhs(0,1)=0;
	feat_mat_lhs(1,1)=-1;
	// 3rd feature vector
	feat_mat_rhs(0,0)=1;
	feat_mat_rhs(1,0)=1;
	// 4th feature vector
	feat_mat_rhs(0,1)=-1;
	feat_mat_rhs(1,1)=1;

	// wrap feat_mat in Shogun features
	CDenseFeatures<float64_t>* features_lhs=new CDenseFeatures<float64_t>(feat_mat_lhs);
	CDenseFeatures<float64_t>* features_rhs=new CDenseFeatures<float64_t>(feat_mat_rhs);

	// put features into distance object to compute squared Euclidean distances
	CEuclideanDistance* euclidean=new CEuclideanDistance(features_lhs,features_rhs);
	euclidean->set_disable_sqrt(true);
	euclidean->parallel->set_num_threads(1);

	SGMatrix<float64_t> distance_matrix=euclidean->get_distance_matrix();

	// check distance matrix
	EXPECT_EQ(distance_matrix(0,0), euclidean->distance(0,0));
	EXPECT_EQ(distance_matrix(0,1), euclidean->distance(0,1));
	EXPECT_EQ(distance_matrix(1,0), euclidean->distance(1,0));
	EXPECT_EQ(distance_matrix(1,1), euclidean->distance(1,1));

	// release memory
	SG_UNREF(euclidean); // the features are unref-ed here as well
}
