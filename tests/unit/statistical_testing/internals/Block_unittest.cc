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

#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/features/Features.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/statistical_testing/internals/Block.h>
#include <gtest/gtest.h>

using namespace shogun;
using namespace internal;

TEST(Block, create_blocks)
{
	const index_t dim=3;
	const index_t num_vec=8;
	const index_t blocksize=2;

	SGMatrix<float64_t> data_p(dim, num_vec);
	std::iota(data_p.matrix, data_p.matrix+dim*num_vec, 0);

	using feat_type=CDenseFeatures<float64_t>;
	auto feats_p=new feat_type(data_p);

	// check whether correct number of blocks has been formed
	auto blocks=Block::create_blocks(feats_p, num_vec/blocksize, blocksize);
	ASSERT_TRUE(blocks.size()==size_t(num_vec/blocksize));

	// check const cast operator
	for (auto it=blocks.begin(); it!=blocks.end(); ++it)
	{
		const Block& block=*it;
		auto block_feats=static_cast<const CFeatures*>(block);
		ASSERT_TRUE(block_feats->get_num_vectors()==blocksize);
	}

	// check non-const cast operator
	for (auto it=blocks.begin(); it!=blocks.end(); ++it)
	{
		Block& block=*it;
		auto block_feats=static_cast<std::shared_ptr<CFeatures>>(block);
		ASSERT_TRUE(block_feats->get_num_vectors()==blocksize);
	}

	// check const get() method
	for (auto it=blocks.begin(); it!=blocks.end(); ++it)
	{
		const Block& block=*it;
		auto block_feats=block.get();
		ASSERT_TRUE(block_feats->get_num_vectors()==blocksize);
	}

	// check non-const get() method
	for (auto it=blocks.begin(); it!=blocks.end(); ++it)
	{
		Block& block=*it;
		auto block_feats=block.get();
		ASSERT_TRUE(block_feats->get_num_vectors()==blocksize);
	}

	// check for proper block-wise organizing
	SGVector<index_t> inds(blocksize);
	std::iota(inds.vector, inds.vector+inds.vlen, 0);
	for (size_t i=0; i<blocks.size(); ++i)
	{
		feats_p->add_subset(inds);
		SGMatrix<float64_t> subset=feats_p->get_feature_matrix();
		SGMatrix<float64_t> blockd=static_cast<feat_type*>(blocks[i].get())->get_feature_matrix();
		ASSERT_TRUE(subset.equals(blockd));
		feats_p->remove_subset();
		std::for_each(inds.vector, inds.vector+inds.vlen, [&blocksize](index_t& val) { val+=blocksize; });
	}

	// no clean-up should be required
}
