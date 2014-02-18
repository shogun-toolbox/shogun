/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 Parijat Mazumdar
 */

#include <gtest/gtest.h>
#include <shogun/mathematics/Math.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/preprocessor/PCA.h>

using namespace shogun;

TEST(PCA, PCA_output_test)
{
	SGMatrix<float64_t> data(2,3);
	data(0,0)=1.0*cos(M_PI/3.0);
	data(0,1)=2.0*cos(M_PI/3.0);
	data(0,2)=3.0*cos(M_PI/3.0);
	data(1,0)=1.0*sin(M_PI/3.0);
	data(1,1)=2.0*sin(M_PI/3.0);
	data(1,2)=3.0*sin(M_PI/3.0);

	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(data);
	CPCA* pca=new CPCA();
	pca->set_target_dim(1);
	pca->init(features);
	
	SGMatrix<float64_t> mat=pca->apply_to_feature_matrix(features);

	EXPECT_EQ(1,mat.num_rows);
	EXPECT_EQ(3,mat.num_cols);

	EXPECT_NEAR(-1,mat(0,0),0.0000001);
	EXPECT_NEAR(0,mat(0,1),0.0000001);
	EXPECT_NEAR(1,mat(0,2),0.0000001);

	SG_UNREF(pca);
	SG_UNREF(features);
}
