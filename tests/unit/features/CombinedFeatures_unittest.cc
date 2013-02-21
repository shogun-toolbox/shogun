/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012-2013 Heiko Strathmann
 */

#include <shogun/base/init.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/CombinedFeatures.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(CombinedFeaturesTest,create_merged_copy)
{
	/* create two matrices, feature objects for them, call create_merged_copy,
	 * and check if it worked */

	index_t n_1=3;
	index_t n_2=4;
	index_t dim=2;

	SGMatrix<float64_t> data_1(dim,n_1);
	for (index_t i=0; i<dim*n_1; ++i)
		data_1.matrix[i]=i;

//	data_1.display_matrix("data_1");

	SGMatrix<float64_t> data_2(dim,n_2);
	for (index_t i=0; i<dim*n_2; ++i)
		data_2.matrix[i]=CMath::randn_double();

//	data_1.display_matrix("data_2");

	CCombinedFeatures* features_1=new CCombinedFeatures();
	CCombinedFeatures* features_2=new CCombinedFeatures();

	features_1->append_feature_obj(new CDenseFeatures<float64_t>(data_1));
	features_2->append_feature_obj(new CDenseFeatures<float64_t>(data_2));

	CFeatures* concatenation=features_1->create_merged_copy(features_2);

	CFeatures* sub=((CCombinedFeatures*)concatenation)->get_first_feature_obj();
	CDenseFeatures<float64_t>* casted_sub=
			dynamic_cast<CDenseFeatures<float64_t>*>(sub);
	ASSERT(casted_sub);
	SGMatrix<float64_t> concat_data=casted_sub->get_feature_matrix();
	SG_UNREF(sub);
//	concat_data.display_matrix("concat_data");

	/* check for equality with data_1 */
	for (index_t i=0; i<dim*n_1; ++i)
		EXPECT_EQ(data_1.matrix[i], concat_data.matrix[i]);

	/* check for equality with data_2 */
	for (index_t i=0; i<dim*n_2; ++i)
		EXPECT_EQ(data_2.matrix[i], concat_data.matrix[n_1*dim+i]);

	SG_UNREF(concatenation);
	SG_UNREF(features_1);
	SG_UNREF(features_2);
}
