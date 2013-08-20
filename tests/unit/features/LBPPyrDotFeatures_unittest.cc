/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Evangelos Anagnostopoulos
 */

#include <shogun/base/init.h>
#include <shogun/features/LBPPyrDotFeatures.h>
#include <shogun/lib/SGMatrix.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(LBPPyrDotFeatures, dot_test)
{
	float64_t correct_result = 73.0; // computed from original code

	int32_t image_width = 10;
	int32_t image_height = 10;
	int32_t num_pyrs = 5;
	int32_t img_len = image_width * image_height;
	int32_t num_vecs = 2;
	SGMatrix<uint32_t> mat(img_len, num_vecs);
	for (index_t i=0; i<num_vecs; i++)
	{
		for (index_t j=0; j<img_len; j++)
			mat(j,i) = j % image_height + i;
	}

	CDenseFeatures<uint32_t>* dense_feat = new CDenseFeatures<uint32_t>(mat);
	CLBPPyrDotFeatures* lbp_feat = new CLBPPyrDotFeatures(dense_feat, image_width, 
			image_height, num_pyrs);

	float64_t result = lbp_feat->dot(0, lbp_feat, 1);
	EXPECT_EQ(result, correct_result);

	SG_UNREF(lbp_feat);
}

TEST(LBPPyrDotFeatures, dense_dot_test)
{
	float64_t correct_result = 9.0; // computed from original code

	int32_t image_width = 10;
	int32_t image_height = 10;
	int32_t num_pyrs = 5;
	int32_t img_len = image_width * image_height;
	int32_t num_vecs = 2;
	SGMatrix<uint32_t> mat(img_len, num_vecs);
	for (index_t j=0; j<img_len; j++)
		mat(j,0) = j % image_height;

	for (index_t j=0; j<img_len; j++)
		mat(j,1) = j % 20 + 1; 

	CDenseFeatures<uint32_t>* dense_feat = new CDenseFeatures<uint32_t>(mat);
	CLBPPyrDotFeatures* lbp_feat = new CLBPPyrDotFeatures(dense_feat, image_width, 
			image_height, num_pyrs);

	SGVector<char> vec = lbp_feat->get_transformed_image(1);
	SGVector<float64_t> tmp(vec.vlen);
	for (index_t i=0; i<vec.vlen; i++)
		tmp[i] = vec[i];
	float64_t result = lbp_feat->dense_dot(0, tmp.vector, vec.vlen);
	EXPECT_EQ(result, correct_result);

	SG_UNREF(lbp_feat);
}

TEST(LBPPyrDotFeatures, add_to_dense_test)
{
	int32_t image_width = 10;
	int32_t image_height = 10;
	int32_t num_pyrs = 5;
	int32_t img_len = image_width * image_height;
	int32_t num_vecs = 2;
	SGMatrix<uint32_t> mat(img_len, num_vecs);
	for (index_t j=0; j<img_len; j++)
		mat(j,0) = j % image_height;

	for (index_t j=0; j<img_len; j++)
		mat(j,1) = j % 20 + 1; 

	CDenseFeatures<uint32_t>* dense_feat = new CDenseFeatures<uint32_t>(mat);
	CLBPPyrDotFeatures* lbp_feat = new CLBPPyrDotFeatures(dense_feat, image_width, 
			image_height, num_pyrs);

	SGVector<char> tmp1 = lbp_feat->get_transformed_image(0);
	SGVector<char> tmp2 = lbp_feat->get_transformed_image(1);
	SGVector<float64_t> vec1(tmp1.vlen);
	SGVector<float64_t> vec2(tmp2.vlen);
	SGVector<float64_t> vec3(tmp2.vlen);
	for (index_t i=0; i<vec1.vlen; i++)
	{
		vec1[i] = tmp1[i];
		vec2[i] = tmp2[i];
		vec3[i] = tmp2[i];
	}

	lbp_feat->add_to_dense_vec(1, 0, vec3.vector, vec3.vlen, false);

	for (index_t i=0; i<vec3.vlen; i++)
		EXPECT_EQ(vec3[i], vec2[i] + vec1[i]);

	SG_UNREF(lbp_feat);
}
