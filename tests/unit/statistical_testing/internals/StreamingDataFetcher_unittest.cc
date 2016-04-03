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
#include <algorithm>
#include <shogun/lib/SGMatrix.h>
#include <shogun/features/Features.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/streaming/StreamingDenseFeatures.h>
#include <shogun/statistical_testing/internals/StreamingDataFetcher.h>
#include <gtest/gtest.h>

using namespace shogun;
using namespace internal;

TEST(StreamingDataFetcher, full_data)
{
	const index_t dim=3;
	const index_t num_vec=8;

	SGMatrix<float64_t> data_p(dim, num_vec);
	std::iota(data_p.matrix, data_p.matrix+dim*num_vec, 0);

	using feat_type=CDenseFeatures<float64_t>;
	auto feats_p=new feat_type(data_p);
	CStreamingFeatures *streaming_p = new CStreamingDenseFeatures<float64_t>(feats_p);
	SG_REF(streaming_p); // TODO check why this refcount is required

	StreamingDataFetcher fetcher(streaming_p);
	fetcher.set_num_samples(num_vec);

	fetcher.start();
	auto curr=fetcher.next();
	ASSERT_TRUE(curr!=nullptr);

	auto tmp=dynamic_cast<feat_type*>(curr.get());
	ASSERT_TRUE(tmp!=nullptr);

	curr=fetcher.next();
	ASSERT_TRUE(curr==nullptr);
	fetcher.end();
}

TEST(StreamingDataFetcher, block_data)
{
	const index_t dim=3;
	const index_t num_vec=8;
	const index_t blocksize=2;
	const index_t num_blocks_per_burst=2;

	SGMatrix<float64_t> data_p(dim, num_vec);
	std::iota(data_p.matrix, data_p.matrix+dim*num_vec, 0);

	using feat_type=CDenseFeatures<float64_t>;
	auto feats_p=new feat_type(data_p);
	CStreamingFeatures *streaming_p = new CStreamingDenseFeatures<float64_t>(feats_p);
	SG_REF(streaming_p); // TODO check why this refcount is required

	StreamingDataFetcher fetcher(streaming_p);
	fetcher.set_num_samples(num_vec);

	fetcher.fetch_blockwise()
		.with_blocksize(blocksize)
		.with_num_blocks_per_burst(num_blocks_per_burst);

	fetcher.start();
	auto curr=fetcher.next();
	ASSERT_TRUE(curr!=nullptr);
	while (curr!=nullptr)
	{
		auto tmp=dynamic_cast<feat_type*>(curr.get());
		ASSERT_TRUE(tmp!=nullptr);
		ASSERT_TRUE(tmp->get_num_vectors()==blocksize*num_blocks_per_burst);
		curr=fetcher.next();
	}
	fetcher.end();
}

TEST(StreamingDataFetcher, reset_functionality)
{
	const index_t dim=3;
	const index_t num_vec=8;
	const index_t blocksize=2;
	const index_t num_blocks_per_burst=2;

	SGMatrix<float64_t> data_p(dim, num_vec);
	std::iota(data_p.matrix, data_p.matrix+dim*num_vec, 0);

	using feat_type=CDenseFeatures<float64_t>;
	auto feats_p=new feat_type(data_p);
	CStreamingFeatures *streaming_p = new CStreamingDenseFeatures<float64_t>(feats_p);
	SG_REF(streaming_p); // TODO check why this refcount is required

	StreamingDataFetcher fetcher(streaming_p);
	fetcher.set_num_samples(num_vec);

	fetcher.start();
	auto curr=fetcher.next();
	ASSERT_TRUE(curr!=nullptr);

	auto tmp=dynamic_cast<feat_type*>(curr.get());
	ASSERT_TRUE(tmp!=nullptr);

	curr=fetcher.next();
	ASSERT_TRUE(curr==nullptr);

	fetcher.reset();
	fetcher.fetch_blockwise()
		.with_blocksize(blocksize)
		.with_num_blocks_per_burst(num_blocks_per_burst);

	fetcher.start();
	curr=fetcher.next();
	ASSERT_TRUE(curr!=nullptr);
	while (curr!=nullptr)
	{
		tmp=dynamic_cast<feat_type*>(curr.get());
		ASSERT_TRUE(tmp!=nullptr);
		ASSERT_TRUE(tmp->get_num_vectors()==blocksize*num_blocks_per_burst);
		curr=fetcher.next();
	}
	fetcher.end();
}
