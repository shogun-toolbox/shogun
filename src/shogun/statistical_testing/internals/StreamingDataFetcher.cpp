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
#include <shogun/io/SGIO.h>
#include <shogun/features/Features.h>
#include <shogun/features/streaming/StreamingFeatures.h>
#include <shogun/statistical_testing/internals/StreamingDataFetcher.h>
#include <shogun/statistical_testing/internals/BlockwiseDetails.h>

using namespace shogun;
using namespace internal;

StreamingDataFetcher::StreamingDataFetcher(CStreamingFeatures* samples)
: DataFetcher(), parser_running(false)
{
	REQUIRE(samples!=nullptr, "Samples cannot be null!\n");
	m_samples=samples;
	SG_REF(m_samples);
	m_num_samples=0;
}

StreamingDataFetcher::~StreamingDataFetcher()
{
	end();
	SG_UNREF(m_samples);
}

void StreamingDataFetcher::set_num_samples(index_t num_samples)
{
	m_num_samples=num_samples;
}

void StreamingDataFetcher::shuffle_features()
{
}

void StreamingDataFetcher::unshuffle_features()
{
}

void StreamingDataFetcher::use_fold(index_t i)
{
}

void StreamingDataFetcher::init_active_subset()
{
}

index_t StreamingDataFetcher::get_num_samples() const
{
	if (train_test_mode)
	{
		if (train_mode)
			return m_num_samples*train_test_ratio/(train_test_ratio+1);
		else
			return m_num_samples/(train_test_ratio+1);
	}
	return m_num_samples;
}

void StreamingDataFetcher::start()
{
	REQUIRE(get_num_samples()>0, "Number of samples is not set! It is MANDATORY for streaming features!\n");
	if (m_block_details.m_full_data || m_block_details.m_blocksize>get_num_samples())
	{
		SG_SINFO("Fetching entire data (%d samples)!\n", get_num_samples());
		m_block_details.with_blocksize(get_num_samples());
	}
	m_block_details.m_total_num_blocks=get_num_samples()/m_block_details.m_blocksize;
	m_block_details.m_next_block_index=0;
	if (!parser_running)
	{
		m_samples->start_parser();
		parser_running=true;
	}
}

CFeatures* StreamingDataFetcher::next()
{
	CFeatures* next_samples=nullptr;
	// figure out how many samples to fetch in this burst
	auto num_already_fetched=m_block_details.m_next_block_index*m_block_details.m_blocksize;
	auto num_more_samples=get_num_samples()-num_already_fetched;
	if (num_more_samples>0)
	{
		auto num_samples_this_burst=std::min(m_block_details.m_max_num_samples_per_burst, num_more_samples);
		next_samples=m_samples->get_streamed_features(num_samples_this_burst);
		SG_REF(next_samples);
		m_block_details.m_next_block_index+=m_block_details.m_num_blocks_per_burst;
	}
	return next_samples;
}

void StreamingDataFetcher::reset()
{
	m_block_details.m_next_block_index=0;
	m_samples->reset_stream();
}

void StreamingDataFetcher::end()
{
	if (parser_running)
	{
		m_samples->end_parser();
		parser_running=false;
	}
}
