/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012-2013 Evangelos Anagnostopoulos
 */

#include <shogun/base/init.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/SparseFeatures.h>
#include <shogun/features/CombinedDotFeatures.h>
#include <gtest/gtest.h>
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

	CCombinedDotFeatures* comb_feat = new CCombinedDotFeatures();
	CDenseFeatures<float64_t>* feat_1 = new CDenseFeatures<float64_t>(data_1);
	CDenseFeatures<float64_t>* feat_2 = new CDenseFeatures<float64_t>(data_2);
	CDenseFeatures<float64_t>* feat_3 = new CDenseFeatures<float64_t>(data_3);

	if (comb_feat->append_feature_obj(feat_1))
		EXPECT_EQ(comb_feat->get_num_feature_obj(),1);

	if (comb_feat->append_feature_obj(feat_2))
		EXPECT_EQ(comb_feat->get_num_feature_obj(),2);

	if (comb_feat->insert_feature_obj(feat_3, 1))
		EXPECT_EQ(comb_feat->get_num_feature_obj(),3);

	comb_feat->delete_feature_obj(0);
	EXPECT_EQ(comb_feat->get_num_feature_obj(),2);

	CDenseFeatures<float64_t>* f_1 = (CDenseFeatures<float64_t>*) comb_feat->get_feature_obj(0);
	SGMatrix<float64_t> m_1 = f_1->get_feature_matrix();
	CDenseFeatures<float64_t>* f_2 = (CDenseFeatures<float64_t>*) comb_feat->get_feature_obj(1);
	SGMatrix<float64_t> m_2 = f_2->get_feature_matrix();
	for (index_t i=0; i < 6; i++)
	{
		EXPECT_EQ(m_1[i], data_3[i]);
		EXPECT_EQ(m_2[i], data_2[i]);
	}
	SG_UNREF(f_1);
	SG_UNREF(f_2);

	SG_UNREF(comb_feat);
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

	CCombinedDotFeatures* comb_feat_1 = new CCombinedDotFeatures();
	CCombinedDotFeatures* comb_feat_2 = new CCombinedDotFeatures();
	CDenseFeatures<float64_t>* feat_1 = new CDenseFeatures<float64_t>(data_1);
	CDenseFeatures<float64_t>* feat_2 = new CDenseFeatures<float64_t>(data_2);
	CDenseFeatures<float64_t>* feat_3 = new CDenseFeatures<float64_t>(data_3);

	comb_feat_1->append_feature_obj(feat_1);
	comb_feat_1->append_feature_obj(feat_2);
	comb_feat_1->append_feature_obj(feat_3);
	comb_feat_2->append_feature_obj(feat_3);
	comb_feat_2->append_feature_obj(feat_1);
	comb_feat_2->append_feature_obj(feat_2);

	SG_SINFO("Beginning dot() testing");
	int32_t result = comb_feat_1->dot(0, comb_feat_2, 0);
	EXPECT_EQ(result, -5);
	result = comb_feat_1->dot(1,comb_feat_2,1);
	EXPECT_EQ(result, -50);
	result = comb_feat_1->dot(0,comb_feat_2,1);
	EXPECT_EQ(result, -14);
	SG_SINFO("Completed dot() testing");

	SG_SINFO("Beginning dense_dot() testing");
	float64_t* vector = new float64_t[9];
	for (index_t i=0; i<9; i++)
		vector[i] = 10 + i;

	result = comb_feat_1->dense_dot(1, vector, 9);
	EXPECT_EQ(result, 376);
	SG_SINFO("Completed dense_dot() testing");

	delete [] vector;
	SG_UNREF(comb_feat_1);
	SG_UNREF(comb_feat_2);
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

	CCombinedDotFeatures* comb_feat = new CCombinedDotFeatures();
	CSparseFeatures<float64_t>* feat_1 = new CSparseFeatures<float64_t>(data_1);
	CSparseFeatures<float64_t>* feat_2 = new CSparseFeatures<float64_t>(data_2);
	CSparseFeatures<float64_t>* feat_3 = new CSparseFeatures<float64_t>(data_3);
	comb_feat->append_feature_obj(feat_1);
	comb_feat->append_feature_obj(feat_2);
	comb_feat->append_feature_obj(feat_3);

	EXPECT_EQ(comb_feat->get_nnz_features_for_vector(0), 6);

	float64_t value=0;
	int32_t index=0;
	index_t nnz_index=0;
	void* itcomb = comb_feat->get_feature_iterator(0);
	while (comb_feat->get_next_feature(index, value, itcomb))
		ASSERT_EQ(nnz[nnz_index++], value);

	comb_feat->free_feature_iterator(itcomb);
	SG_UNREF(comb_feat);
}
