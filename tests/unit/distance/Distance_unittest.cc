/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Pan Deng, Fernando Iglesias
 */

#include <gtest/gtest.h>

#include <shogun/distance/CustomMahalanobisDistance.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/lib/SGMatrix.h>

using namespace shogun;

TEST(Distance, custom_mahalanobis)
{
	// Create a couple of simple 2D features
	SGMatrix<float64_t> feat_mat(2,2);
	// First feature vector
	feat_mat[0] = 1.0, feat_mat[1] = 1.0;
	// Second feature vector
	feat_mat[2] = 2.0, feat_mat[3] = 2.0;
	// Create features object
	auto feats = std::make_shared<DenseFeatures<float64_t>>(feat_mat);

	// Create a simple Mahalanobis matrix, equal to the identity
	SGMatrix<float64_t> eye_mat = SGMatrix<float64_t>::create_identity_matrix(2, 1.0);
	auto distance = std::make_shared<CustomMahalanobisDistance>(feats, feats, eye_mat);

	// The distance is equal to zero between equal points
	EXPECT_EQ(distance->distance(0,0), 0.0);
	EXPECT_EQ(distance->distance(1,1), 0.0);
	// The distance is symmetric
	EXPECT_EQ(distance->distance(0,1), distance->distance(1,0));
	// Compare with the actual value computed a priori
	EXPECT_EQ(distance->distance(0,1), 2.0);


}
