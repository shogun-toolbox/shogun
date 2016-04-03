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
#include <type_traits>
#include <shogun/lib/common.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/features/Features.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/statistical_testing/internals/DataManager.h>
#include <gtest/gtest.h>

using namespace shogun;
using namespace internal;

TEST(InitPerFeature, assignment_and_cast_operators)
{
	const index_t dim=1;
	const index_t num_vec=1;
	const index_t num_distributions=1;

	SGMatrix<float64_t> data_p(dim, num_vec);
	data_p(0, 0)=0;
	auto feats_p=new CDenseFeatures<float64_t>(data_p);

	DataManager data_mgr(num_distributions);
	data_mgr.samples_at(0)=feats_p;
	const DataManager& const_data_mgr=data_mgr;

	auto stored_feats=data_mgr.samples_at(0);
	bool typecheck=std::is_same<InitPerFeature, decltype(stored_feats)>::value;
	ASSERT_TRUE(typecheck);
	ASSERT_TRUE(feats_p==stored_feats);

	auto stored_feats2=const_data_mgr.samples_at(0);
	typecheck=std::is_same<CFeatures*, decltype(stored_feats2)>::value;
	ASSERT_TRUE(typecheck);
	ASSERT_TRUE(feats_p==stored_feats2);

	const CFeatures* samples=static_cast<const CFeatures*>(stored_feats);
	ASSERT_TRUE(feats_p==samples);
}
