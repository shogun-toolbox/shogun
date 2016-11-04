/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Fernando J. Iglesias Garcia
 * Written (W) 2016 Soumyajit De
 */

#include <shogun/distance/EuclideanDistance.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/DenseSubSamplesFeatures.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/base/init.h>
#include <gtest/gtest.h>

using namespace shogun;

CDenseFeatures<float64_t>* create_lhs()
{
	// create object with two feature vectors,
	// each column represents a feature vector
	SGMatrix<float64_t> feat_mat_lhs(2,2);
	// 1st feature vector
	feat_mat_lhs(0,0)=0;
	feat_mat_lhs(1,0)=0;
	// 2nd feature vector
	feat_mat_lhs(0,1)=0;
	feat_mat_lhs(1,1)=-1;
	return new CDenseFeatures<float64_t>(feat_mat_lhs);
}

CDenseFeatures<float64_t>* create_rhs()
{
	// create object with two feature vectors,
	// each column represents a feature vector
	SGMatrix<float64_t> feat_mat_rhs(2,2);
	// 3rd feature vector
	feat_mat_rhs(0,0)=1;
	feat_mat_rhs(1,0)=1;
	// 4th feature vector
	feat_mat_rhs(0,1)=-1;
	feat_mat_rhs(1,1)=1;

	return new CDenseFeatures<float64_t>(feat_mat_rhs);
}

TEST(EuclideanDistance,distance)
{
	CDenseFeatures<float64_t>* features_lhs=create_lhs();
	CDenseFeatures<float64_t>* features_rhs=create_rhs();

	// put features into distance object to compute squared Euclidean distances
	CEuclideanDistance* euclidean=new CEuclideanDistance(features_lhs,features_rhs);
	euclidean->set_disable_sqrt(true);

	// check distances computed one by one
	EXPECT_EQ(euclidean->distance(0,0), 2);
	EXPECT_EQ(euclidean->distance(0,1), 2);
	EXPECT_EQ(euclidean->distance(1,0), 5);
	EXPECT_EQ(euclidean->distance(1,1), 5);

	SG_UNREF(euclidean);
}

TEST(EuclideanDistance, distance_precomputed_norms)
{
	CDenseFeatures<float64_t>* features_lhs=create_lhs();
	CDenseFeatures<float64_t>* features_rhs=create_rhs();

	CEuclideanDistance* euclidean=new CEuclideanDistance(features_lhs,features_rhs);
	euclidean->set_disable_sqrt(true);
	euclidean->precompute_lhs();
	euclidean->precompute_rhs();

	// check distances computed one by one
	EXPECT_EQ(euclidean->distance(0,0), 2);
	EXPECT_EQ(euclidean->distance(0,1), 2);
	EXPECT_EQ(euclidean->distance(1,0), 5);
	EXPECT_EQ(euclidean->distance(1,1), 5);

	euclidean->reset_precompute();

	SG_UNREF(euclidean);
}

TEST(EuclideanDistance,get_distance_matrix)
{
	CDenseFeatures<float64_t>* features_lhs=create_lhs();
	CDenseFeatures<float64_t>* features_rhs=create_rhs();

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

	SG_UNREF(euclidean);
}

TEST(EuclideanDistance, heterogenous_features)
{
	CDenseFeatures<float64_t>* features_lhs=create_lhs();
	CDenseFeatures<float64_t>* features_rhs=create_rhs();
	SG_REF(features_lhs);
	SG_REF(features_rhs);

	SGVector<int32_t> idx(2);
	idx[0]=0;
	idx[1]=1;

	CDenseSubSamplesFeatures<float64_t>* subsample_lhs=new CDenseSubSamplesFeatures<float64_t>(features_lhs, idx);
	CDenseSubSamplesFeatures<float64_t>* subsample_rhs=new CDenseSubSamplesFeatures<float64_t>(features_rhs, idx);
	SG_REF(subsample_lhs);
	SG_REF(subsample_rhs);

	CEuclideanDistance* euclidean=new CEuclideanDistance();
	euclidean->set_disable_sqrt(true);
	euclidean->parallel->set_num_threads(1);

	float64_t accuracy=1E-15;

	euclidean->init(subsample_lhs, features_rhs);

	EXPECT_NEAR(euclidean->distance(0, 0), 2, accuracy);
	EXPECT_NEAR(euclidean->distance(0, 1), 2, accuracy);
	EXPECT_NEAR(euclidean->distance(1, 0), 5, accuracy);
	EXPECT_NEAR(euclidean->distance(1, 1), 5, accuracy);

	euclidean->init(features_lhs, subsample_rhs);

	EXPECT_NEAR(euclidean->distance(0, 0), 2, accuracy);
	EXPECT_NEAR(euclidean->distance(0, 1), 2, accuracy);
	EXPECT_NEAR(euclidean->distance(1, 0), 5, accuracy);
	EXPECT_NEAR(euclidean->distance(1, 1), 5, accuracy);

	SG_UNREF(euclidean);
	SG_UNREF(subsample_lhs);
	SG_UNREF(subsample_rhs);
	SG_UNREF(features_lhs);
	SG_UNREF(features_rhs);
}
