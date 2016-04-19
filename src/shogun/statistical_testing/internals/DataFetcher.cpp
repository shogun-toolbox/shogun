/*
 * Restructuring Shogun's statistical hypothesis testing framework.
 * Copyright (C) 2016  Soumyajit De
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http:/www.gnu.org/licenses/>.
 */

#include <algorithm>
#include <shogun/features/Features.h>
#include <shogun/statistical_testing/internals/DataFetcher.h>
#include <shogun/statistical_testing/internals/FeaturesUtil.h>

using namespace shogun;
using namespace internal;

DataFetcher::DataFetcher() : m_num_samples(0), m_samples(nullptr)
{
}

DataFetcher::DataFetcher(CFeatures* samples) : m_samples(samples)
{
	REQUIRE(m_samples!=nullptr, "Samples cannot be null!\n");
	SG_REF(m_samples);
	m_num_samples=m_samples->get_num_vectors();
}

DataFetcher::~DataFetcher()
{
	end();
	SG_UNREF(m_samples);
}

const char* DataFetcher::get_name() const
{
	return "DataFetcher";
}

void DataFetcher::start()
{
	REQUIRE(m_num_samples>0, "Number of samples is 0!\n");
	if (m_block_details.m_blocksize==0)
	{
		SG_SINFO("Block details not set! Fetching entire data (%d samples)!\n", m_num_samples);
		m_block_details.with_blocksize(m_num_samples);
	}
	m_block_details.m_total_num_blocks=m_num_samples/m_block_details.m_blocksize;
	reset();
}

CFeatures* DataFetcher::next()
{
	CFeatures* next_samples=nullptr;
	// figure out how many samples to fetch in this burst
	auto num_already_fetched=m_block_details.m_next_block_index*m_block_details.m_blocksize;
	auto num_more_samples=m_num_samples-num_already_fetched;
	if (num_more_samples>0)
	{
		auto num_samples_this_burst=std::min(m_block_details.m_max_num_samples_per_burst, num_more_samples);
		// create a shallow copy and add proper index subset
		next_samples=FeaturesUtil::create_shallow_copy(m_samples);
		SGVector<index_t> inds(num_samples_this_burst);
		std::iota(inds.vector, inds.vector+inds.vlen, num_already_fetched);
		next_samples->add_subset(inds);

		m_block_details.m_next_block_index+=m_block_details.m_num_blocks_per_burst;
	}
	return next_samples;
}

void DataFetcher::reset()
{
	m_block_details.m_next_block_index=0;
}

void DataFetcher::end()
{
}

const index_t DataFetcher::get_num_samples() const
{
	return m_num_samples;
}

BlockwiseDetails& DataFetcher::fetch_blockwise()
{
	return m_block_details;
}
