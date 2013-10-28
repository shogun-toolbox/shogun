/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Fernando J. Iglesias Garcia
 * Copyright (C) 2013 Fernando J. Iglesias Garcia
 */

#ifdef HAVE_EIGEN3

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
	CDenseFeatures<float64_t>* feats = new CDenseFeatures<float64_t>(feat_mat);

	// Create a simple Mahalanobis matrix, equal to the identity
	SGMatrix<float64_t> eye_mat = SGMatrix<float64_t>::create_identity_matrix(2, 1.0);
	CCustomMahalanobisDistance* distance = new CCustomMahalanobisDistance(feats, feats, eye_mat);

	// The distance is equal to zero between equal points
	EXPECT_EQ(distance->distance(0,0), 0.0);
	EXPECT_EQ(distance->distance(1,1), 0.0);
	// The distance is symmetric
	EXPECT_EQ(distance->distance(0,1), distance->distance(1,0));
	// Compare with the actual value computed a priori
	EXPECT_EQ(distance->distance(0,1), 2.0);

	SG_UNREF(distance)
}

#endif /* HAVE_EIGEN3 */
