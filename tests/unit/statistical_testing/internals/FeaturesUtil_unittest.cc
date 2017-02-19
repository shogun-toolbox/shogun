/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2016 Soumyajit De
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
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

#include <algorithm>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/features/Features.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/statistical_testing/internals/FeaturesUtil.h>
#include <gtest/gtest.h>

using namespace shogun;
using namespace internal;

TEST(FeaturesUtil, create_shallow_copy)
{
	const index_t dim=2;
	const index_t num_vec=10;

	SGMatrix<float64_t> data(dim, num_vec);
	std::iota(data.matrix, data.matrix+dim*num_vec, 0);

	auto feats=new CDenseFeatures<float64_t>(data);
	SGVector<index_t> inds(5);
	std::iota(inds.data(), inds.data()+inds.size(), 3);
	feats->add_subset(inds);
	SGVector<index_t> inds2(2);
	std::iota(inds2.data(), inds2.data()+inds2.size(), 1);
	feats->add_subset(inds2);

	auto shallow_copy=static_cast<CDenseFeatures<float64_t>*>(FeaturesUtil::create_shallow_copy(feats));
	int32_t num_feats=0, num_vecs=0;
	float64_t* copied_data=shallow_copy->get_feature_matrix(num_feats, num_vecs);
	ASSERT_TRUE(data.data()==copied_data);
	ASSERT_TRUE(dim==num_feats);
	ASSERT_TRUE(num_vec==num_vecs);

	SGMatrix<float64_t> src=feats->get_feature_matrix();
	SGMatrix<float64_t> dst=shallow_copy->get_feature_matrix();
	ASSERT(src.equals(dst));

	shallow_copy->remove_all_subsets();
	SG_UNREF(shallow_copy);

	feats->remove_all_subsets();
	SG_UNREF(feats);
}

TEST(FeaturesUtil, create_merged_copy)
{
	const index_t dim=2;
	const index_t num_vec=3;

	SGMatrix<float64_t> data(dim, num_vec);
	std::iota(data.matrix, data.matrix+dim*num_vec, 0);

	auto feats_a=new CDenseFeatures<float64_t>(data);
	SGVector<index_t> inds_a(2);
	inds_a[0]=1;
	inds_a[1]=2;
	feats_a->add_subset(inds_a);
	SGMatrix<float64_t> data_a=feats_a->get_feature_matrix();

	auto feats_b=new CDenseFeatures<float64_t>(data);
	SGVector<index_t> inds_b(2);
	inds_b[0]=0;
	inds_b[1]=2;
	feats_b->add_subset(inds_b);
	SGMatrix<float64_t> data_b=feats_b->get_feature_matrix();

	SGMatrix<float64_t> merged(dim, data_a.num_cols+data_b.num_cols);
	std::copy(data_a.data(), data_a.data()+data_a.size(), merged.data());
	std::copy(data_b.data(), data_b.data()+data_b.size(), merged.data()+data_a.size());

	auto merged_copy=static_cast<CDenseFeatures<float64_t>*>(FeaturesUtil::create_merged_copy(feats_a, feats_b));
	SGMatrix<float64_t> copied(merged_copy->get_feature_matrix());
	ASSERT_TRUE(merged.equals(copied));

	SG_UNREF(merged_copy);
	SG_UNREF(feats_a);
	SG_UNREF(feats_b);
}

TEST(FeaturesUtil, clone_subset_stack)
{
	const index_t dim=2;
	const index_t num_vec=10;

	SGMatrix<float64_t> data(dim, num_vec);
	std::iota(data.matrix, data.matrix+dim*num_vec, 0);

	auto feats=new CDenseFeatures<float64_t>(data);
	SGVector<index_t> inds(5);
	std::iota(inds.data(), inds.data()+inds.size(), 3);
	feats->add_subset(inds);
	SGVector<index_t> inds2(2);
	std::iota(inds2.data(), inds2.data()+inds2.size(), 1);
	feats->add_subset(inds2);

	auto copy=new CDenseFeatures<float64_t>(data);
	FeaturesUtil::clone_subset_stack(feats, copy);

	auto src_subset_stack=feats->get_subset_stack();
	auto dst_subset_stack=copy->get_subset_stack();
	ASSERT_TRUE(src_subset_stack->equals(dst_subset_stack));
	SG_UNREF(src_subset_stack);
	SG_UNREF(dst_subset_stack);

	copy->remove_all_subsets();
	SG_UNREF(copy);

	feats->remove_all_subsets();
	SG_UNREF(feats);
}
