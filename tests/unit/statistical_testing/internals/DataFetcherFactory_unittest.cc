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

#include <memory>
#include <cstring>
#include <shogun/lib/SGMatrix.h>
#include <shogun/features/Features.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/streaming/StreamingDenseFeatures.h>
#include <shogun/statistical_testing/internals/DataFetcher.h>
#include <shogun/statistical_testing/internals/StreamingDataFetcher.h>
#include <shogun/statistical_testing/internals/DataFetcherFactory.h>
#include <gtest/gtest.h>

using namespace shogun;
using namespace internal;

TEST(DataFetcherFactory, get_instance)
{
	const index_t dim=1;
	const index_t num_vec=1;

	SGMatrix<float64_t> data_p(dim, num_vec);
	data_p(0, 0)=0;

	using feat_type=CDenseFeatures<float64_t>;
	auto feats_p=new feat_type(data_p);

	std::unique_ptr<DataFetcher> fetcher(DataFetcherFactory::get_instance(feats_p));
	ASSERT_TRUE(strcmp(fetcher->get_name(), "DataFetcher")==0);

	CStreamingDenseFeatures<float64_t> *streaming_p=new CStreamingDenseFeatures<float64_t>(feats_p);
	SG_REF(streaming_p);

	std::unique_ptr<DataFetcher> streaming_fetcher(DataFetcherFactory::get_instance(streaming_p));
	ASSERT_TRUE(strcmp(streaming_fetcher->get_name(), "StreamingDataFetcher")==0);
}
