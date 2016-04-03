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
#include <shogun/features/streaming/StreamingFeatures.h>
#include <shogun/statistical_testing/internals/StreamingDataFetcher.h>


using namespace shogun;
using namespace internal;

StreamingDataFetcher::StreamingDataFetcher(CStreamingFeatures* samples) : DataFetcher(), parser_running(false)
{
	SG_REF(samples);
	m_samples = std::shared_ptr<CStreamingFeatures>(samples, [](CFeatures* ptr) { SG_UNREF(ptr); });
	m_num_samples = 0;
}

StreamingDataFetcher::~StreamingDataFetcher()
{
	end();
}

const char* StreamingDataFetcher::get_name() const
{
	return "StreamingDataFetcher";
}

void StreamingDataFetcher::set_num_samples(index_t num_samples)
{
	m_num_samples = num_samples;
}

void StreamingDataFetcher::start()
{
	ASSERT(m_num_samples);
	if (m_block_details.m_blocksize == 0)
	{
		m_block_details.with_blocksize(m_num_samples);
	}
	m_block_details.m_total_num_blocks = m_num_samples / m_block_details.m_blocksize;
	m_block_details.m_next_block_index = 0;
	if (!parser_running)
	{
		m_samples->start_parser();
		parser_running = true;
		// TODO check if resetting the stream is required
	}
}

std::shared_ptr<CFeatures> StreamingDataFetcher::next()
{
	auto num_more_samples = m_num_samples - m_block_details.m_next_block_index * m_block_details.m_blocksize;
	if (num_more_samples > 0)
	{
		auto num_samples_this_burst = m_block_details.m_max_num_samples_per_burst;
		if (num_samples_this_burst > num_more_samples)
		{
			num_samples_this_burst = num_more_samples;
		}

		CFeatures* streamed = m_samples->get_streamed_features(num_samples_this_burst);
		m_block_details.m_next_block_index += m_block_details.m_num_blocks_per_burst;
		return std::shared_ptr<CFeatures>(streamed, [](CFeatures* ptr) { SG_UNREF(ptr); });
	}
	return nullptr;
}

void StreamingDataFetcher::reset()
{
	m_block_details.m_next_block_index = 0;
	m_samples->reset_stream();
}

void StreamingDataFetcher::end()
{
	if (parser_running)
	{
		m_samples->end_parser();
		parser_running = false;
	}
}
