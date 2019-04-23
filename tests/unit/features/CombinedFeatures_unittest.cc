/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Thoralf Klein, Heiko Strathmann, Bjoern Esser,
 *          Evangelos Anagnostopoulos
 */

#include <gtest/gtest.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/CombinedFeatures.h>
#include <shogun/mathematics/NormalDistribution.h>

#include <random>

using namespace shogun;

TEST(CombinedFeaturesTest,test_array_operations)
{
	index_t n_1=3;
	index_t dim=2;

	SGMatrix<float64_t> data_1(dim,n_1);
	SGMatrix<float64_t> data_2(dim,n_1);
	SGMatrix<float64_t> data_3(dim,n_1);
	for (index_t i=0; i < dim*n_1; i++)
	{
		data_1[i] = i;
		data_2[i] = -i;
		data_3[i] = 2*i;
	}

	auto comb_feat = std::make_shared<CombinedFeatures>();
	auto feat_1 = std::make_shared<DenseFeatures<float64_t>>(data_1);
	auto feat_2 = std::make_shared<DenseFeatures<float64_t>>(data_2);
	auto feat_3 = std::make_shared<DenseFeatures<float64_t>>(data_3);

	if (comb_feat->append_feature_obj(feat_1))
		EXPECT_EQ(comb_feat->get_num_feature_obj(),1);

	if (comb_feat->append_feature_obj(feat_2))
		EXPECT_EQ(comb_feat->get_num_feature_obj(),2);

	if (comb_feat->insert_feature_obj(feat_3, 1))
		EXPECT_EQ(comb_feat->get_num_feature_obj(),3);

	comb_feat->delete_feature_obj(0);
	EXPECT_EQ(comb_feat->get_num_feature_obj(),2);

	auto f_1 = comb_feat->get_feature_obj(0)->as<DenseFeatures<float64_t>>();
	SGMatrix<float64_t> m_1 = f_1->get_feature_matrix();
	auto f_2 = comb_feat->get_feature_obj(1)->as<DenseFeatures<float64_t>>();
	SGMatrix<float64_t> m_2 = f_2->get_feature_matrix();
	for (index_t i=0; i < dim*n_1; i++)
	{
		EXPECT_EQ(m_1[i], data_3[i]);
		EXPECT_EQ(m_2[i], data_2[i]);
	}




}

TEST(CombinedFeaturesTest,create_merged_copy)
{
	/* create two matrices, feature objects for them, call create_merged_copy,
	 * and check if it worked */
	int32_t seed = 100;
	index_t n_1=3;
	index_t n_2=4;
	index_t dim=2;

	SGMatrix<float64_t> data_1(dim,n_1);
	for (index_t i=0; i<dim*n_1; ++i)
		data_1.matrix[i]=i;

//	data_1.display_matrix("data_1");
	std::mt19937_64 prng(seed);
	NormalDistribution<float64_t> normal_dist;
	SGMatrix<float64_t> data_2(dim,n_2);
	for (index_t i=0; i<dim*n_2; ++i)
		data_2.matrix[i]=normal_dist(prng);

//	data_1.display_matrix("data_2");

	auto features_1=std::make_shared<CombinedFeatures>();
	auto features_2=std::make_shared<CombinedFeatures>();

	features_1->append_feature_obj(std::make_shared<DenseFeatures<float64_t>>(data_1));
	features_2->append_feature_obj(std::make_shared<DenseFeatures<float64_t>>(data_2));

	auto concatenation=features_1->create_merged_copy(features_2);

	auto sub=concatenation->as<CombinedFeatures>()->get_first_feature_obj();
	auto casted_sub=sub->as<DenseFeatures<float64_t>>();
	ASSERT(casted_sub);
	SGMatrix<float64_t> concat_data=casted_sub->get_feature_matrix();
//	concat_data.display_matrix("concat_data");

	/* check for equality with data_1 */
	for (index_t i=0; i<dim*n_1; ++i)
		EXPECT_EQ(data_1.matrix[i], concat_data.matrix[i]);

	/* check for equality with data_2 */
	for (index_t i=0; i<dim*n_2; ++i)
		EXPECT_EQ(data_2.matrix[i], concat_data.matrix[n_1*dim+i]);




}
