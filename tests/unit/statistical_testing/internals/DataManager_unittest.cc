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

#include <vector>
#include <algorithm>
#include <shogun/lib/SGMatrix.h>
#include <shogun/features/Features.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/streaming/StreamingDenseFeatures.h>
#include <shogun/statistical_testing/internals/DataManager.h>
#include <shogun/statistical_testing/internals/NextSamples.h>
#include <gtest/gtest.h>

using namespace shogun;
using namespace internal;

TEST(DataManager, full_data_one_distribution_normal_feats)
{
	const index_t dim=3;
	const index_t num_vec=8;
	const index_t num_distributions=1;

	SGMatrix<float64_t> data_p(dim, num_vec);
	std::iota(data_p.matrix, data_p.matrix+dim*num_vec, 0);

	auto feats_p=new CDenseFeatures<float64_t>(data_p);

	DataManager mgr(num_distributions);
	mgr.samples_at(0)=feats_p;

	mgr.start();

	auto next_burst=mgr.next();
	ASSERT_TRUE(!next_burst.empty());
	ASSERT_TRUE(next_burst.num_blocks()==1);

	auto tmp=dynamic_cast<CDenseFeatures<float64_t>*>(next_burst[0][0].get());
	ASSERT_TRUE(tmp!=nullptr);
	ASSERT_TRUE(tmp->get_num_vectors()==num_vec);

	next_burst=mgr.next();
	ASSERT_TRUE(next_burst.empty());

	mgr.end();
}

TEST(DataManager, full_data_one_distribution_streaming_feats)
{
	const index_t dim=3;
	const index_t num_vec=8;
	const index_t num_distributions=1;

	SGMatrix<float64_t> data_p(dim, num_vec);
	std::iota(data_p.matrix, data_p.matrix+dim*num_vec, 0);

	auto feats_p=new CDenseFeatures<float64_t>(data_p);
	auto streaming_p=new CStreamingDenseFeatures<float64_t>(feats_p);

	DataManager mgr(num_distributions);
	mgr.samples_at(0)=streaming_p;
	mgr.num_samples_at(0)=num_vec;

	mgr.start();

	auto next_burst=mgr.next();
	ASSERT_TRUE(!next_burst.empty());
	ASSERT_TRUE(next_burst.num_blocks()==1);

	auto tmp=dynamic_cast<CDenseFeatures<float64_t>*>(next_burst[0][0].get());
	ASSERT_TRUE(tmp!=nullptr);
	ASSERT_TRUE(tmp->get_num_vectors()==num_vec);

	next_burst=mgr.next();
	ASSERT_TRUE(next_burst.empty());

	mgr.end();
}

TEST(DataManager, full_data_two_distributions_normal_feats)
{
	const index_t dim=3;
	const index_t num_vec=8;
	const index_t num_distributions=2;

	SGMatrix<float64_t> data_p(dim, num_vec);
	std::iota(data_p.matrix, data_p.matrix+dim*num_vec, 0);

	SGMatrix<float64_t> data_q(dim, num_vec);
	std::iota(data_q.matrix, data_q.matrix+dim*num_vec, dim*num_vec);

	using feat_type=CDenseFeatures<float64_t>;
	auto feats_p=new feat_type(data_p);
	auto feats_q=new feat_type(data_q);

	DataManager mgr(num_distributions);
	mgr.samples_at(0)=feats_p;
	mgr.samples_at(1)=feats_q;

	mgr.start();

	auto next_burst=mgr.next();
	ASSERT_TRUE(!next_burst.empty());
	ASSERT_TRUE(next_burst.num_blocks()==1);

	auto tmp_p=dynamic_cast<feat_type*>(next_burst[0][0].get());
	auto tmp_q=dynamic_cast<feat_type*>(next_burst[1][0].get());

	ASSERT_TRUE(tmp_p!=nullptr);
	ASSERT_TRUE(tmp_q!=nullptr);
	ASSERT_TRUE(tmp_p->get_num_vectors()==num_vec);
	ASSERT_TRUE(tmp_q->get_num_vectors()==num_vec);

	next_burst=mgr.next();
	ASSERT_TRUE(next_burst.empty());
}

TEST(DataManager, full_data_two_distributions_streaming_feats)
{
	const index_t dim=3;
	const index_t num_vec=8;
	const index_t num_distributions=2;

	SGMatrix<float64_t> data_p(dim, num_vec);
	std::iota(data_p.matrix, data_p.matrix+dim*num_vec, 0);

	SGMatrix<float64_t> data_q(dim, num_vec);
	std::iota(data_q.matrix, data_q.matrix+dim*num_vec, dim*num_vec);

	using feat_type=CDenseFeatures<float64_t>;
	auto feats_p=new feat_type(data_p);
	auto feats_q=new feat_type(data_q);
	auto streaming_p=new CStreamingDenseFeatures<float64_t>(feats_p);
	auto streaming_q=new CStreamingDenseFeatures<float64_t>(feats_q);

	DataManager mgr(num_distributions);
	mgr.samples_at(0)=streaming_p;
	mgr.samples_at(1)=streaming_q;
	mgr.num_samples_at(0)=num_vec;
	mgr.num_samples_at(1)=num_vec;

	mgr.start();

	auto next_burst=mgr.next();
	ASSERT_TRUE(!next_burst.empty());
	ASSERT_TRUE(next_burst.num_blocks()==1);

	auto tmp_p=dynamic_cast<feat_type*>(next_burst[0][0].get());
	auto tmp_q=dynamic_cast<feat_type*>(next_burst[1][0].get());

	ASSERT_TRUE(tmp_p!=nullptr);
	ASSERT_TRUE(tmp_q!=nullptr);
	ASSERT_TRUE(tmp_p->get_num_vectors()==num_vec);
	ASSERT_TRUE(tmp_q->get_num_vectors()==num_vec);

	next_burst=mgr.next();
	ASSERT_TRUE(next_burst.empty());
}

TEST(DataManager, block_data_one_distribution_normal_feats)
{
	const index_t dim=3;
	const index_t num_vec=8;
	const index_t blocksize=2;
	const index_t num_blocks_per_burst=2;
	const index_t num_distributions=1;

	SGMatrix<float64_t> data_p(dim, num_vec);
	std::iota(data_p.matrix, data_p.matrix+dim*num_vec, 0);

	auto feats_p=new CDenseFeatures<float64_t>(data_p);

	DataManager mgr(num_distributions);
	mgr.samples_at(0)=feats_p;
	mgr.set_blocksize(blocksize);
	mgr.set_num_blocks_per_burst(num_blocks_per_burst);

	mgr.start();

	auto next_burst=mgr.next();
	ASSERT_TRUE(!next_burst.empty());

	auto total=0;

	while (!next_burst.empty())
	{
		ASSERT_TRUE(next_burst.num_blocks()==num_blocks_per_burst);
		for (auto i=0; i<next_burst.num_blocks(); ++i)
		{
			auto tmp=dynamic_cast<CDenseFeatures<float64_t>*>(next_burst[0][i].get());
			ASSERT_TRUE(tmp!=nullptr);
			ASSERT_TRUE(tmp->get_num_vectors()==blocksize);
			total+=tmp->get_num_vectors();
		}
		next_burst=mgr.next();
	}
	ASSERT_TRUE(total==num_vec);
}

TEST(DataManager, block_data_one_distribution_streaming_feats)
{
	const index_t dim=3;
	const index_t num_vec=8;
	const index_t blocksize=2;
	const index_t num_blocks_per_burst=2;
	const index_t num_distributions=1;

	SGMatrix<float64_t> data_p(dim, num_vec);
	std::iota(data_p.matrix, data_p.matrix+dim*num_vec, 0);

	auto feats_p=new CDenseFeatures<float64_t>(data_p);
	auto streaming_p=new CStreamingDenseFeatures<float64_t>(feats_p);

	DataManager mgr(num_distributions);
	mgr.samples_at(0)=streaming_p;
	mgr.num_samples_at(0)=num_vec;
	mgr.set_blocksize(blocksize);
	mgr.set_num_blocks_per_burst(num_blocks_per_burst);

	mgr.start();

	auto next_burst=mgr.next();
	ASSERT_TRUE(!next_burst.empty());

	auto total=0;

	while (!next_burst.empty())
	{
		ASSERT_TRUE(next_burst.num_blocks()==num_blocks_per_burst);
		for (auto i=0; i<next_burst.num_blocks(); ++i)
		{
			auto tmp=dynamic_cast<CDenseFeatures<float64_t>*>(next_burst[0][i].get());
			ASSERT_TRUE(tmp!=nullptr);
			ASSERT_TRUE(tmp->get_num_vectors()==blocksize);
			total+=tmp->get_num_vectors();
		}
		next_burst=mgr.next();
	}
	ASSERT_TRUE(total==num_vec);
}

TEST(DataManager, block_data_two_distributions_normal_feats_equal_blocksize)
{
	const index_t dim=3;
	const index_t num_vec=8;
	const index_t blocksize=2;
	const index_t num_blocks_per_burst=2;
	const index_t num_distributions=2;

	SGMatrix<float64_t> data_p(dim, num_vec);
	std::iota(data_p.matrix, data_p.matrix+dim*num_vec, 0);

	SGMatrix<float64_t> data_q(dim, num_vec);
	std::iota(data_q.matrix, data_q.matrix+dim*num_vec, dim*num_vec);

	using feat_type=CDenseFeatures<float64_t>;
	auto feats_p=new feat_type(data_p);
	auto feats_q=new feat_type(data_q);

	DataManager mgr(num_distributions);
	mgr.samples_at(0)=feats_p;
	mgr.samples_at(1)=feats_q;
	mgr.set_blocksize(blocksize);
	mgr.set_num_blocks_per_burst(num_blocks_per_burst);

	mgr.start();

	auto next_burst=mgr.next();
	ASSERT_TRUE(!next_burst.empty());

	auto total=0;

	while (!next_burst.empty())
	{
		ASSERT_TRUE(next_burst.num_blocks()==num_blocks_per_burst);
		for (auto i=0; i<next_burst.num_blocks(); ++i)
		{
			auto tmp_p=dynamic_cast<feat_type*>(next_burst[0][i].get());
			auto tmp_q=dynamic_cast<feat_type*>(next_burst[1][i].get());
			ASSERT_TRUE(tmp_p!=nullptr);
			ASSERT_TRUE(tmp_q!=nullptr);
			ASSERT_TRUE(tmp_p->get_num_vectors()==blocksize/2);
			ASSERT_TRUE(tmp_q->get_num_vectors()==blocksize/2);
			total+=tmp_p->get_num_vectors();
		}
		next_burst=mgr.next();
	}
	ASSERT_TRUE(total==num_vec);
}

TEST(DataManager, block_data_two_distributions_streaming_feats_equal_blocksize)
{
	const index_t dim=3;
	const index_t num_vec=8;
	const index_t blocksize=2;
	const index_t num_blocks_per_burst=2;
	const index_t num_distributions=2;

	SGMatrix<float64_t> data_p(dim, num_vec);
	std::iota(data_p.matrix, data_p.matrix+dim*num_vec, 0);

	SGMatrix<float64_t> data_q(dim, num_vec);
	std::iota(data_q.matrix, data_q.matrix+dim*num_vec, dim*num_vec);

	using feat_type=CDenseFeatures<float64_t>;
	auto feats_p=new feat_type(data_p);
	auto feats_q=new feat_type(data_q);
	auto streaming_p=new CStreamingDenseFeatures<float64_t>(feats_p);
	auto streaming_q=new CStreamingDenseFeatures<float64_t>(feats_q);

	DataManager mgr(num_distributions);
	mgr.samples_at(0)=streaming_p;
	mgr.samples_at(1)=streaming_q;
	mgr.num_samples_at(0)=num_vec;
	mgr.num_samples_at(1)=num_vec;
	mgr.set_blocksize(blocksize);
	mgr.set_num_blocks_per_burst(num_blocks_per_burst);

	mgr.start();

	auto next_burst=mgr.next();
	ASSERT_TRUE(!next_burst.empty());

	auto total=0;

	while (!next_burst.empty())
	{
		ASSERT_TRUE(next_burst.num_blocks()==num_blocks_per_burst);
		for (auto i=0; i<next_burst.num_blocks(); ++i)
		{
			auto tmp_p=dynamic_cast<feat_type*>(next_burst[0][i].get());
			auto tmp_q=dynamic_cast<feat_type*>(next_burst[1][i].get());
			ASSERT_TRUE(tmp_p!=nullptr);
			ASSERT_TRUE(tmp_q!=nullptr);
			ASSERT_TRUE(tmp_p->get_num_vectors()==blocksize/2);
			ASSERT_TRUE(tmp_q->get_num_vectors()==blocksize/2);
			total+=tmp_p->get_num_vectors();
		}
		next_burst=mgr.next();
	}
	ASSERT_TRUE(total==num_vec);
}

TEST(DataManager, block_data_two_distributions_normal_feats_different_blocksize)
{
	const index_t dim=3;
	const index_t num_vec_p=8;
	const index_t num_vec_q=12;
	const index_t blocksize=5;
	const index_t num_blocks_per_burst=3;
	const index_t num_distributions=2;

	auto blocksize_p=blocksize*num_vec_p/(num_vec_p+num_vec_q);
	auto blocksize_q=blocksize*num_vec_q/(num_vec_p+num_vec_q);

	SGMatrix<float64_t> data_p(dim, num_vec_p);
	std::iota(data_p.matrix, data_p.matrix+dim*num_vec_p, 0);

	SGMatrix<float64_t> data_q(dim, num_vec_q);
	std::iota(data_q.matrix, data_q.matrix+dim*num_vec_q, dim*num_vec_p);

	using feat_type=CDenseFeatures<float64_t>;
	auto feats_p=new feat_type(data_p);
	auto feats_q=new feat_type(data_q);

	DataManager mgr(num_distributions);
	mgr.samples_at(0)=feats_p;
	mgr.samples_at(1)=feats_q;
	mgr.set_blocksize(blocksize);
	mgr.set_num_blocks_per_burst(num_blocks_per_burst);

	mgr.start();

	auto next_burst=mgr.next();
	ASSERT_TRUE(!next_burst.empty());

	auto total_p=0;
	auto total_q=0;

	while (!next_burst.empty())
	{
		for (auto i=0; i<next_burst.num_blocks(); ++i)
		{
			auto tmp_p=dynamic_cast<feat_type*>(next_burst[0][i].get());
			auto tmp_q=dynamic_cast<feat_type*>(next_burst[1][i].get());
			ASSERT_TRUE(tmp_p!=nullptr);
			ASSERT_TRUE(tmp_q!=nullptr);
			ASSERT_TRUE(tmp_p->get_num_vectors()==blocksize_p);
			ASSERT_TRUE(tmp_q->get_num_vectors()==blocksize_q);
			total_p+=tmp_p->get_num_vectors();
			total_q+=tmp_q->get_num_vectors();
		}
		next_burst=mgr.next();
	}
	ASSERT_TRUE(total_p==num_vec_p);
	ASSERT_TRUE(total_q==num_vec_q);
}

TEST(DataManager, block_data_two_distributions_streaming_feats_different_blocksize)
{
	const index_t dim=3;
	const index_t num_vec_p=8;
	const index_t num_vec_q=12;
	const index_t blocksize=5;
	const index_t num_blocks_per_burst=3;
	const index_t num_distributions=2;

	auto blocksize_p=blocksize*num_vec_p/(num_vec_p+num_vec_q);
	auto blocksize_q=blocksize*num_vec_q/(num_vec_p+num_vec_q);

	SGMatrix<float64_t> data_p(dim, num_vec_p);
	std::iota(data_p.matrix, data_p.matrix+dim*num_vec_p, 0);

	SGMatrix<float64_t> data_q(dim, num_vec_q);
	std::iota(data_q.matrix, data_q.matrix+dim*num_vec_q, dim*num_vec_p);

	using feat_type=CDenseFeatures<float64_t>;
	auto feats_p=new feat_type(data_p);
	auto feats_q=new feat_type(data_q);
	auto streaming_p=new CStreamingDenseFeatures<float64_t>(feats_p);
	auto streaming_q=new CStreamingDenseFeatures<float64_t>(feats_q);

	DataManager mgr(num_distributions);
	mgr.samples_at(0)=streaming_p;
	mgr.samples_at(1)=streaming_q;
	mgr.num_samples_at(0)=num_vec_p;
	mgr.num_samples_at(1)=num_vec_q;
	mgr.set_blocksize(blocksize);
	mgr.set_num_blocks_per_burst(num_blocks_per_burst);

	mgr.start();

	auto next_burst=mgr.next();
	ASSERT_TRUE(!next_burst.empty());

	auto total_p=0;
	auto total_q=0;

	while (!next_burst.empty())
	{
		for (auto i=0; i<next_burst.num_blocks(); ++i)
		{
			auto tmp_p=dynamic_cast<feat_type*>(next_burst[0][i].get());
			auto tmp_q=dynamic_cast<feat_type*>(next_burst[1][i].get());
			ASSERT_TRUE(tmp_p!=nullptr);
			ASSERT_TRUE(tmp_q!=nullptr);
			ASSERT_TRUE(tmp_p->get_num_vectors()==blocksize_p);
			ASSERT_TRUE(tmp_q->get_num_vectors()==blocksize_q);
			total_p+=tmp_p->get_num_vectors();
			total_q+=tmp_q->get_num_vectors();
		}
		next_burst=mgr.next();
	}
	ASSERT_TRUE(total_p==num_vec_p);
	ASSERT_TRUE(total_q==num_vec_q);
}
