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

#include <shogun/features/Features.h>
#include <shogun/statistical_testing/internals/NextSamples.h>

using namespace shogun;
using namespace internal;

NextSamples::NextSamples(index_t num_distributions) : m_num_blocks(0)
{
	next_samples.resize(num_distributions);
}

NextSamples& NextSamples::operator=(const NextSamples& other)
{
	clear();
	m_num_blocks=other.m_num_blocks;
	next_samples=other.next_samples;
	return *this;
}

NextSamples::~NextSamples()
{
	clear();
}

std::vector<Block>& NextSamples::operator[](size_t i)
{
	REQUIRE(i>=0 && i<next_samples.size(),
			"index (%d) must be between [0,%d]!\n",
			i, next_samples.size()-1);
	return next_samples[i];
}

const std::vector<Block>& NextSamples::operator[](size_t i) const
{
	REQUIRE(i>=0 && i<next_samples.size(),
			"index (%d) must be between [0,%d]!\n",
			i, next_samples.size()-1);
	return next_samples[i];
}

const index_t NextSamples::num_blocks() const
{
	return m_num_blocks;
}

const bool NextSamples::empty() const
{
	typedef const std::vector<Block> type;
	return std::any_of(next_samples.cbegin(), next_samples.cend(), [](type& f) { return f.size()==0; });
}

void NextSamples::clear()
{
	typedef std::vector<Block> type;
	std::for_each(next_samples.begin(), next_samples.end(), [](type& f) { f.clear(); });
	next_samples.clear();
}
