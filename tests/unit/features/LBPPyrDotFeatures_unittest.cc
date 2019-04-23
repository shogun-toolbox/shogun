/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Thoralf Klein, Bjoern Esser, Evangelos Anagnostopoulos
 */

#include <gtest/gtest.h>
#include <shogun/features/LBPPyrDotFeatures.h>
#include <shogun/lib/SGMatrix.h>

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

	auto dense_feat = std::make_shared<DenseFeatures<uint32_t>>(mat);
	auto lbp_feat = std::make_shared<LBPPyrDotFeatures>(dense_feat, image_width,
			image_height, num_pyrs);

	float64_t result = lbp_feat->dot(0, lbp_feat, 1);
	EXPECT_EQ(result, correct_result);


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

	auto dense_feat = std::make_shared<DenseFeatures<uint32_t>>(mat);
	auto lbp_feat = std::make_shared<LBPPyrDotFeatures>(dense_feat, image_width,
			image_height, num_pyrs);

	SGVector<char> vec = lbp_feat->get_transformed_image(1);
	SGVector<float64_t> tmp(vec.vlen);
	for (index_t i=0; i<vec.vlen; i++)
		tmp[i] = vec[i];
	float64_t result = lbp_feat->dot(0, tmp);
	EXPECT_EQ(result, correct_result);


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

	auto dense_feat = std::make_shared<DenseFeatures<uint32_t>>(mat);
	auto lbp_feat = std::make_shared<LBPPyrDotFeatures>(dense_feat, image_width,
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


}
