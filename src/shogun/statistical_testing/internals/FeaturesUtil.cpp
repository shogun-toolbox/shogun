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

#include <stack>
#include <algorithm>
#include <shogun/io/SGIO.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/features/Features.h>
#include <shogun/features/FeatureTypes.h>
#include <shogun/features/Subset.h>
#include <shogun/features/SubsetStack.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/statistical_testing/internals/FeaturesUtil.h>

using namespace shogun;
using namespace internal;

CFeatures* FeaturesUtil::create_shallow_copy(CFeatures* other)
{
	SG_SDEBUG("Entering!\n");
	CFeatures* shallow_copy=nullptr;
	if (other->get_feature_type()==F_DREAL && other->get_feature_class()==C_DENSE)
	{
		auto casted=static_cast<CDenseFeatures<float64_t>*>(other);

		// use the same underlying feature matrix, no ref-count
		int32_t num_feats=0, num_vecs=0;
		float64_t* data=casted->get_feature_matrix(num_feats, num_vecs);
		SG_SDEBUG("Using underlying feature matrix with %d dimensions and %d feature vectors!\n", num_feats, num_vecs);
		SGMatrix<float64_t> feats_matrix(data, num_feats, num_vecs, false);
		shallow_copy=new CDenseFeatures<float64_t>(feats_matrix);

		// clone the subsets if there are any
		CSubsetStack* src_subset_stack=casted->get_subset_stack();
		if (src_subset_stack->has_subsets())
		{
			SG_SDEBUG("Subset present, cloning the subsets!\n");
			CSubsetStack* subset_stack=static_cast<CSubsetStack*>(src_subset_stack->clone());
			std::stack<SGVector<index_t>> stack;
			while (subset_stack->has_subsets())
			{
				stack.push(subset_stack->get_last_subset()->get_subset_idx());
				subset_stack->remove_subset();
			}
			SG_UNREF(subset_stack);
			while (!stack.empty())
			{
				shallow_copy->add_subset(stack.top());
				stack.pop();
			}
		}
		SG_UNREF(src_subset_stack);
	}
	else
		SG_SNOTIMPLEMENTED;
	SG_SDEBUG("Leaving!\n");
	return shallow_copy;
}

CFeatures* FeaturesUtil::create_merged_copy(CFeatures* feats_a, CFeatures* feats_b)
{
	SG_SDEBUG("Entering!\n");
	REQUIRE(feats_a->get_feature_type()==feats_b->get_feature_type(),
			"The feature types of the underlying feature objects should be same!\n");
	REQUIRE(feats_a->get_feature_class()==feats_b->get_feature_class(),
			"The feature classes of the underlying feature objects should be same!\n");

	CFeatures* merged_copy=nullptr;

	if (feats_a->get_feature_type()==F_DREAL && feats_a->get_feature_class()==C_DENSE)
	{
		auto casted_a=static_cast<CDenseFeatures<float64_t>*>(feats_a);
		auto casted_b=static_cast<CDenseFeatures<float64_t>*>(feats_b);

		REQUIRE(casted_a->get_num_features()==casted_b->get_num_features(),
				"The number of features from a (%d) has to be equal with that of b (%d)!\n",
				casted_a->get_num_features(), casted_b->get_num_features());

		SGMatrix<float64_t> data_a=casted_a->get_feature_matrix();
		SGMatrix<float64_t> data_b=casted_b->get_feature_matrix();
		ASSERT(data_a.num_rows==data_b.num_rows);

		SGMatrix<float64_t> merged(data_a.num_rows, data_a.num_cols+data_b.num_cols);
		std::copy(data_a.data(), data_a.data()+data_a.size(), merged.data());
		std::copy(data_b.data(), data_b.data()+data_b.size(), merged.data()+data_a.size());

		merged_copy=new CDenseFeatures<float64_t>(merged);
	}
	else
		SG_SNOTIMPLEMENTED;

	SG_SDEBUG("Leaving!\n");
	return merged_copy;
}
