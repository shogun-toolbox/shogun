/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2016 - 2017 Soumyajit De
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
#include <numeric>
#include <shogun/lib/SGVector.h>
#include <shogun/features/Features.h>
#include <shogun/statistical_testing/internals/Block.h>

using namespace shogun;
using namespace internal;

Block::Block(std::shared_ptr<Features> feats, index_t index, index_t size) : m_feats(feats)
{
	require(m_feats!=nullptr, "Underlying feature object cannot be null!");

	// create a shallow copy and subset current block separately
	auto block=m_feats->shallow_subset_copy();

	SGVector<index_t> inds(size);
	std::iota(inds.vector, inds.vector+inds.vlen, index*size);
	block->add_subset(inds);

	m_block=block;
}

Block::Block(const Block& other) : m_block(other.m_block), m_feats(other.m_feats)
{
}

Block& Block::operator=(const Block& other)
{
	m_block=other.m_block;
	m_feats=other.m_feats;

	return *this;
}

Block::~Block()
{


}

std::vector<Block> Block::create_blocks(std::shared_ptr<Features> feats, index_t num_blocks, index_t size)
{
	std::vector<Block> vec;
	for (index_t i=0; i<num_blocks; ++i)
		vec.push_back(Block(feats, i, size));
	return vec;
}
