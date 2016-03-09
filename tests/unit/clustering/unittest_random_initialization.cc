/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2016 Manan Tomar
 */

#include <shogun/labels/MulticlassLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/clustering/KMeans.h>
#include <shogun/distance/EuclideanDistance.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(KMeans, random_initialization_test)
{
    /* create a triangle with three points as data set (0,0) (5,5) (10,0) */
    SGMatrix<float64_t> tri(2, 3);
    tri(0,0)=0;
    tri(0,1)=5;
    tri(0,2)=10;
    tri(1,0)=0;
    tri(1,1)=5;
    tri(1,2)=0;

    CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(tri);
    CEuclideanDistance* distance=new CEuclideanDistance(features, features);
    CKMeans* clustering=new CKMeans(2, distance);

    SGVector<int32_t> ClList=SGVector<int32_t>(3);
    ClList.zero();
    SGVector<float64_t> weights_set=SGVector<float64_t>(2);
    weights_set.zero();

    clustering->set_random_centers(weights_set, ClList, 3);

    CDenseFeatures<float64_t>* randomised_centers=CDenseFeatures<float64_t>::obtain_from_generic(distance->get_lhs());
    SGMatrix<float64_t> randomised_centers_matrix=randomised_centers->get_feature_matrix();

    EXPECT_TRUE(!((randomised_centers_matrix(0,0)==randomised_centers_matrix(0,1))&&(randomised_centers_matrix(1,0)==randomised_centers_matrix(1,1))));

    for(int32_t i=0; i<2; i++)
        EXPECT_TRUE(weights_set[i]!=0);

    SG_UNREF(randomised_centers);
    SG_UNREF(clustering);
    SG_UNREF(features);

}
