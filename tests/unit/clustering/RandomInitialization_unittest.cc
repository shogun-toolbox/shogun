/*
* Copyright (c) 2016, Shogun Toolbox Foundation
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:

* 1. Redistributions of source code must retain the above copyright notice,
* this list of conditions and the following disclaimer.
*
* 2. Redistributions in binary form must reproduce the above copyright notice,
* this list of conditions and the following disclaimer in the documentation
* and/or other materials provided with the distribution.
*
* 3. Neither the name of the copyright holder nor the names of its
* contributors may be used to endorse or promote products derived from this
* software without specific prior written permission.

* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
* ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
* LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
* CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
* SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
* INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
* CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
* ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
* POSSIBILITY OF SUCH DAMAGE.
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

    /* vector for storing cluster center of each point */
    SGVector<int32_t> classes=SGVector<int32_t>(3);
    classes.zero();
    /* vector for size of each cluster */
    SGVector<float64_t> cluster_size=SGVector<float64_t>(2);
    cluster_size.zero();

    clustering->set_random_centers(cluster_size, classes, 3);

    CDenseFeatures<float64_t>* randomised_centers=CDenseFeatures<float64_t>::obtain_from_generic(distance->get_lhs());
    SGMatrix<float64_t> randomised_centers_matrix=randomised_centers->get_feature_matrix();

    EXPECT_FALSE((randomised_centers_matrix(0,0)==randomised_centers_matrix(0,1)) &&
            (randomised_centers_matrix(1,0)==randomised_centers_matrix(1,1)));

    for(int32_t i=0; i<2; i++)
        EXPECT_TRUE(cluster_size[i]!=0);

    SG_UNREF(clustering);
    SG_UNREF(features);

}
