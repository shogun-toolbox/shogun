/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Evangelos Anagnostopoulos, Thoralf Klein, Soeren Sonnenburg,
 *          Bjoern Esser, Viktor Gal
 */

#include <gtest/gtest.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/SparseFeatures.h>
#include <shogun/features/CombinedDotFeatures.h>
#include <shogun/io/SGIO.h>

using namespace shogun;

TEST(CombinedDotFeaturesTest, test_array_operations)
{
	SGMatrix<float64_t> data_1(3,2);
	SGMatrix<float64_t> data_2(3,2);
	SGMatrix<float64_t> data_3(3,2);
	for (index_t i=0; i < 6; i++)
	{
		data_1[i] = i;
		data_2[i] = -i;
		data_3[i] = 2*i;
	}

	auto comb_feat = std::make_shared<CombinedDotFeatures>();
	auto feat_1 = std::make_shared<DenseFeatures<float64_t>>(data_1);
	auto feat_2 = std::make_shared<DenseFeatures<float64_t>>(data_2);
	auto feat_3 = std::make_shared<DenseFeatures<float64_t>>(data_3);

	if (comb_feat->append_feature_obj(feat_1))
	{
		EXPECT_EQ(comb_feat->get_num_feature_obj(),1);
	}

	if (comb_feat->append_feature_obj(feat_2))
	{
		EXPECT_EQ(comb_feat->get_num_feature_obj(),2);
	}

	if (comb_feat->insert_feature_obj(feat_3, 1))
	{
		EXPECT_EQ(comb_feat->get_num_feature_obj(),3);
	}

	comb_feat->delete_feature_obj(0);
	EXPECT_EQ(comb_feat->get_num_feature_obj(),2);

	auto f_1 = comb_feat->get_feature_obj(0)->as<DenseFeatures<float64_t>>();
	SGMatrix<float64_t> m_1 = f_1->get_feature_matrix();
	auto f_2 = comb_feat->get_feature_obj(1)->as<DenseFeatures<float64_t>>();
	SGMatrix<float64_t> m_2 = f_2->get_feature_matrix();
	for (index_t i=0; i < 6; i++)
	{
		EXPECT_EQ(m_1[i], data_3[i]);
		EXPECT_EQ(m_2[i], data_2[i]);
	}




}

TEST(CombinedDotFeaturesTest, dot_products)
{
	SGMatrix<float64_t> data_1(3,2);
	SGMatrix<float64_t> data_2(3,2);
	SGMatrix<float64_t> data_3(3,2);
	for (index_t i=0; i < 6; i++)
	{
		data_1[i] = i;
		data_2[i] = -i;
		data_3[i] = 2*i;
	}

	auto comb_feat_1 = std::make_shared<CombinedDotFeatures>();
	auto comb_feat_2 = std::make_shared<CombinedDotFeatures>();
	auto feat_1 = std::make_shared<DenseFeatures<float64_t>>(data_1);
	auto feat_2 = std::make_shared<DenseFeatures<float64_t>>(data_2);
	auto feat_3 = std::make_shared<DenseFeatures<float64_t>>(data_3);

	comb_feat_1->append_feature_obj(feat_1);
	comb_feat_1->set_subfeature_weight(0, 1);
	comb_feat_1->append_feature_obj(feat_2);
	comb_feat_1->set_subfeature_weight(1, 2);
	comb_feat_1->append_feature_obj(feat_3);
	comb_feat_1->set_subfeature_weight(2, 3);

	comb_feat_2->append_feature_obj(feat_3);
	comb_feat_2->set_subfeature_weight(0, 3);
	comb_feat_2->append_feature_obj(feat_1);
	comb_feat_2->set_subfeature_weight(1, 1);
	comb_feat_2->append_feature_obj(feat_2);
	comb_feat_2->set_subfeature_weight(2, 2);

	io::info("Beginning dot() testing");
	int32_t result = comb_feat_1->dot(0, comb_feat_2, 0);
	// comb_feat_1[0] dot comb_feat_2[0] =
	//		feat_1_weight * feat_3_weight * (feat_1 dot feat_3) +
	//		feat_2_weight * feat_1_weight * (feat_2 dot feat_1) +
	//		feat_3_weight * feat_2_weight * (feat_3 dot feat_2) +
	EXPECT_EQ(result, 1*3*10 - 2*1*5 - 3*2*10);
	result = comb_feat_1->dot(1,comb_feat_2,1);
	EXPECT_EQ(result, 1*3*100 - 2*1*50 - 3*2*100);
	result = comb_feat_1->dot(0,comb_feat_2,1);
	EXPECT_EQ(result, 1*3*28 - 2*1*14 - 3*2*28);
	io::info("Completed dot() testing");

	io::info("Beginning dot() testing");
	SGVector<float64_t> vector(9);
	for (index_t i=0; i<9; i++)
	{
		vector[i] = 10 + i;
	}

	result = comb_feat_1->dot(1, vector);
	EXPECT_EQ(result, 1 * 134 - 2 * 170 + 3 * 412);
	io::info("Completed dense_dot() testing");

}

TEST(CombinedDotFeaturesTest, nnz_features)
{
	SGMatrix<float64_t> data_1(3,2);
	SGMatrix<float64_t> data_2(3,2);
	SGMatrix<float64_t> data_3(3,2);
	for (index_t i=0; i < 6; i++)
	{
		data_1[i] = i;
		data_2[i] = -i;
		data_3[i] = 2*i;
	}
	/* the concatenation of the first vector of the matrices gives:
	 * 0, 1, 2, 0, -1, -2, 0, 2, 4
	 * and so the non-zero features are 1, 2, -1, -2, 2, 4.
	 */
	SGVector<float64_t> nnz(6);
	nnz[0] = 1;
	nnz[1] = 2;
	nnz[2] = -1;
	nnz[3] = -2;
	nnz[4] = 2;
	nnz[5] = 4;

	auto comb_feat = std::make_shared<CombinedDotFeatures>();
	auto feat_1 = std::make_shared<SparseFeatures<float64_t>>(data_1);
	auto feat_2 = std::make_shared<SparseFeatures<float64_t>>(data_2);
	auto feat_3 = std::make_shared<SparseFeatures<float64_t>>(data_3);
	comb_feat->append_feature_obj(feat_1);
	comb_feat->append_feature_obj(feat_2);
	comb_feat->append_feature_obj(feat_3);

	EXPECT_EQ(comb_feat->get_nnz_features_for_vector(0), 6);

	float64_t value=0;
	int32_t index=0;
	index_t nnz_index=0;
	void* itcomb = comb_feat->get_feature_iterator(0);
	while (comb_feat->get_next_feature(index, value, itcomb))
	{
		ASSERT_EQ(nnz[nnz_index++], value);
	}

	comb_feat->free_feature_iterator(itcomb);

}

TEST(CombinedDotFeaturesTest, feature_weights)
{
	index_t num_subfeats = 20;
	index_t insert_pos = 10;

	std::vector<std::shared_ptr<DenseFeatures<float64_t>>> feats(num_subfeats);
	for (index_t i = 0; i < num_subfeats; i++)
		feats[i] = std::make_shared<DenseFeatures<float64_t>>();

	auto comb_feat = std::make_shared<CombinedDotFeatures>();
	// test get_subfeature_weight & set_subfeature_weight
	for (index_t i = 0; i < num_subfeats; i++)
	{
		comb_feat->append_feature_obj(feats[i]);
		comb_feat->set_subfeature_weight(i, i);
	}
	auto subfeat_weights = comb_feat->get_subfeature_weights();
	for (index_t i = 0; i < num_subfeats; i++)
	{
		EXPECT_EQ(comb_feat->get_subfeature_weight(i), i);
		EXPECT_EQ(subfeat_weights[i], i);
	}

	// test insert_feature_obj
	auto inserted_feat = std::make_shared<DenseFeatures<float64_t>>();
	comb_feat->insert_feature_obj(inserted_feat, insert_pos);
	comb_feat->set_subfeature_weight(10, -1);
	for (index_t i = 0; i < insert_pos; i++)
	{
		EXPECT_EQ(comb_feat->get_subfeature_weight(i), i);
	}
	EXPECT_EQ(comb_feat->get_subfeature_weight(insert_pos), -1);
	for (index_t i = insert_pos+1; i < num_subfeats; i++)
	{
		EXPECT_EQ(comb_feat->get_subfeature_weight(i), i - 1);
	}
}

TEST(CombinedDotFeaturesTest, dense_dot_range)
{
	index_t num_subfeats = 20;
	index_t num_vectors = 10;
	index_t dim = 10;
	float64_t b = 23;

	auto comb_feat = std::make_shared<CombinedDotFeatures>();

	// first vector is [0, 1, ..., dim-1]
	// second vector is [0, 2,..., 2*dim-2], and so on
	SGMatrix<float64_t> data(dim, num_vectors);
	for (index_t r = 0; r < dim; r++)
		for (index_t c = 0; c < num_vectors; c++)
			data(r, c) = r * (c + 1);

	// the stacked ith vector is the same as the ith vector but
	// repeated num_subfeats times
	for (index_t i = 0; i < num_subfeats; i++)
	{
		comb_feat->append_feature_obj(std::make_shared<DenseFeatures<float64_t>>(data));
		comb_feat->set_subfeature_weight(i, i);
	}

	float64_t* output = new float64_t[num_vectors];
	float64_t* alphas = new float64_t[num_vectors];
	std::iota(alphas, alphas + num_vectors, 0);

	// vec = [0, 1, ..., dim-1] repeated num_subfets times
	float64_t* vec = new float64_t[dim * num_subfeats];
	for (index_t i = 0; i < num_subfeats; i++)
		std::iota(vec + i * dim, vec + (i + 1) * dim, 0);

	comb_feat->dense_dot_range(
	    output, 0, num_vectors, alphas, vec, dim * num_subfeats, b);
	// output[i] = alphas[i] * subfeat_weight_sum * (feature_vector(i) dot vec)
	//			 = alphas[i] * subfeat_weight_sum * sum_{j = 0}^{j = dim-1} j^2
	float64_t sum = (dim / 6.0) * (dim - 1) * (2 * dim - 1);
	float64_t weights_sum = num_subfeats * (num_subfeats - 1) / 2.0;
	for (index_t i = 0; i < num_vectors; i++)
	{
		EXPECT_EQ(output[i], alphas[i] * (i + 1) * weights_sum * sum + b);
	}

	delete[] output;
	delete[] alphas;
	delete[] vec;
}

TEST(CombinedDotFeaturesTest, add_to_dense_vec)
{
	index_t num_subfeats = 20;
	index_t num_vectors = 10;
	index_t dim = 10;
	index_t vec_idx1 = 5;
	float64_t alpha = 3;

	SGMatrix<float64_t> data(dim, num_vectors);
	for (int j = 0; j < num_vectors * dim; j++)
		data[j] = (j % 2 == 0) ? j : -j;

	auto comb_feat = std::make_shared<CombinedDotFeatures>();
	for (index_t i = 0; i < num_subfeats; i++)
	{
		comb_feat->append_feature_obj(std::make_shared<DenseFeatures<float64_t>>(data));
		comb_feat->set_subfeature_weight(i, i);
	}

	float64_t* vec = new float64_t[dim * num_subfeats];
	float64_t* vec2 = new float64_t[dim * num_subfeats];
	std::iota(vec, vec + dim * num_subfeats, 0);
	std::iota(vec2, vec2 + dim * num_subfeats, 0);
	comb_feat->add_to_dense_vec(
	    alpha, vec_idx1, vec, dim * num_subfeats, false);
	comb_feat->add_to_dense_vec(
	    alpha, vec_idx1, vec2, dim * num_subfeats, true);

	for (index_t i = 0; i < dim * num_subfeats; i++)
	{
		float64_t subfeat_weight = comb_feat->get_subfeature_weight(i / dim);
		float64_t feat_val = data(int(i % dim), vec_idx1);
		EXPECT_EQ(vec[i], i + alpha * subfeat_weight * feat_val);
		EXPECT_EQ(vec2[i], i + alpha * subfeat_weight * std::abs(feat_val));
	}

	delete[] vec;
	delete[] vec2;
}
