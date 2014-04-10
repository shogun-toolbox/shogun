/*
* Copyright (c) The Shogun Machine Learning Toolbox
* Written (w) 2014 pl8787
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice, this
* list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright notice,
* this list of conditions and the following disclaimer in the documentation
* and/or other materials provided with the distribution.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
* ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
* The views and conclusions contained in the software and documentation are those
* of the authors and should not be interpreted as representing official policies,
* either expressed or implied, of the Shogun Development Team.
*/

#include <shogun/features/IndexFeatures.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(IndexFeaturesTest,basic_create)
{
	index_t vlen = 10;
	SGVector<index_t> index_vector(vlen);
	index_vector.range_fill();
	CIndexFeatures* index_features = new CIndexFeatures(index_vector);

	EXPECT_EQ(index_features->get_num_vectors(), vlen);
	EXPECT_EQ(index_features->get_feature_class(), C_INDEX);
	EXPECT_EQ(index_features->get_feature_type(), F_INT);

	SGVector<index_t> v_index_feature = index_features->get_feature_index();

	for(index_t i=0; i<vlen; i++)
		EXPECT_EQ(v_index_feature[i], index_vector[i]);

	SG_UNREF(index_features);
}

TEST(IndexFeaturesTest,subset_copy)
{
	index_t vlen = 10;
	SGVector<index_t> index_vector(vlen);
	index_vector.range_fill();
	index_vector.permute();

	SGVector<index_t> sub_idx(vlen/2);
	sub_idx.range_fill();
	sub_idx.permute();

	CIndexFeatures* index_features = new CIndexFeatures(index_vector);

	index_features->add_subset(sub_idx);

	EXPECT_EQ(index_features->get_num_vectors(), vlen/2);
	EXPECT_EQ(index_features->get_feature_class(), C_INDEX);
	EXPECT_EQ(index_features->get_feature_type(), F_INT);

	SGVector<index_t> v_index_feature = index_features->get_feature_index();

	for(index_t i=0; i<vlen/2; i++)
		EXPECT_EQ(v_index_feature[i], index_vector[sub_idx[i]]);

	SG_UNREF(index_features);
}
