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
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <algorithm>
#include <shogun/features/Features.h>
#include <shogun/statistics/experimental/internals/DataFetcher.h>


using namespace shogun;
using namespace internal;

DataFetcher::DataFetcher() : m_num_samples(0)
{
}

DataFetcher::DataFetcher(CFeatures* samples)
{
	SG_REF(samples);
	m_samples = std::shared_ptr<CFeatures>(samples, [](CFeatures* ptr) { SG_UNREF(ptr); });
	m_num_samples = m_samples->get_num_vectors();
}

DataFetcher::~DataFetcher()
{
}

const char* DataFetcher::get_name() const
{
	return "DataFetcher";
}

void DataFetcher::start()
{
	if (m_block_details.m_blocksize == 0)
	{
		m_block_details.with_blocksize(m_num_samples);
	}
	m_block_details.m_total_num_blocks = m_num_samples / m_block_details.m_blocksize;
	reset();
}

std::shared_ptr<CFeatures> DataFetcher::next()
{
	auto num_more_samples = m_num_samples - m_block_details.m_next_block_index * m_block_details.m_blocksize;
	if (num_more_samples > 0)
	{
		auto num_samples_this_burst = m_block_details.m_max_num_samples_per_burst;
		if (num_samples_this_burst > num_more_samples)
		{
			num_samples_this_burst = num_more_samples;
		}
		if (num_samples_this_burst < m_num_samples)
		{
			m_samples->remove_subset();
			SGVector<index_t> inds(num_samples_this_burst);
			std::iota(inds.vector, inds.vector + inds.vlen, m_block_details.m_next_block_index * m_block_details.m_blocksize);
			m_samples->add_subset(inds);
		}

		m_block_details.m_next_block_index += m_block_details.m_num_blocks_per_burst;
		return m_samples;
	}
	return nullptr;
}

void DataFetcher::reset()
{
	m_block_details.m_next_block_index = 0;
	m_samples->remove_all_subsets();
}

void DataFetcher::end()
{
	m_samples->remove_all_subsets();
}

const index_t DataFetcher::get_num_samples() const
{
	return m_num_samples;
}

BlockwiseDetails& DataFetcher::fetch_blockwise()
{
	return m_block_details;
}
