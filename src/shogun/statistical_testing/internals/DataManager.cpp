/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 - 2016 Soumyajit De
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

#include <shogun/io/SGIO.h>
#include <shogun/features/Features.h>
#include <shogun/statistical_testing/internals/DataManager.h>
#include <shogun/statistical_testing/internals/NextSamples.h>
#include <shogun/statistical_testing/internals/DataFetcher.h>
#include <shogun/statistical_testing/internals/DataFetcherFactory.h>

using namespace shogun;
using namespace internal;

DataManager::DataManager(size_t num_distributions)
{
	SG_SDEBUG("Data manager instance initialized with %d data sources!\n", num_distributions);
	fetchers.resize(num_distributions);
	std::fill(fetchers.begin(), fetchers.end(), nullptr);
}

DataManager::~DataManager()
{
}

index_t DataManager::get_num_samples() const
{
	SG_SDEBUG("Entering!\n");
	index_t n=0;
	using fetcher_type=const std::unique_ptr<DataFetcher>;
	if (std::any_of(fetchers.begin(), fetchers.end(), [](fetcher_type& f) { return f->m_num_samples==0; }))
		SG_SERROR("number of samples from all the distributions are not set!")
	else
		std::for_each(fetchers.begin(), fetchers.end(), [&n](fetcher_type& f) { n+=f->m_num_samples; });
	SG_SDEBUG("Leaving!\n");
	return n;
}

index_t DataManager::get_min_blocksize() const
{
	SG_SDEBUG("Entering!\n");
	index_t min_blocksize=0;
	using fetcher_type=const std::unique_ptr<DataFetcher>;
	if (std::any_of(fetchers.begin(), fetchers.end(), [](fetcher_type& f) { return f->m_num_samples==0; }))
		SG_SERROR("number of samples from all the distributions are not set!")
	else
	{
		index_t divisor=0;
		std::function<index_t(index_t, index_t)> gcd=[&gcd](index_t m, index_t n)
		{
			return n==0?m:gcd(n, m%n);
		};
		for (size_t i=0; i<fetchers.size(); ++i)
			divisor=gcd(divisor, fetchers[i]->m_num_samples);
		min_blocksize=get_num_samples()/divisor;
	}
	SG_SDEBUG("min blocksize is %d!", min_blocksize);
	SG_SDEBUG("Leaving!\n");
	return min_blocksize;
}

void DataManager::set_blocksize(index_t blocksize)
{
	SG_SDEBUG("Entering!\n");
	auto n=get_num_samples();

	REQUIRE(n>0,
			"Total number of samples is 0! Please set the number of samples!\n");
	REQUIRE(blocksize>0 && blocksize<=n,
			"The blocksize has to be within [0, %d], given = %d!\n",
			n, blocksize);
	REQUIRE(n%blocksize==0,
			"Total number of samples (%d) has to be divisble by the blocksize (%d)!\n",
			n, blocksize);

	for (size_t i=0; i<fetchers.size(); ++i)
	{
		index_t m=fetchers[i]->m_num_samples;
		REQUIRE((blocksize*m)%n==0,
				"Blocksize (%d) cannot be even distributed with a ratio of %f!\n",
				blocksize, m/n);
		fetchers[i]->fetch_blockwise().with_blocksize(blocksize*m/n);
		SG_SDEBUG("block[%d].size = ", i, blocksize*m/n);
	}
	SG_SDEBUG("Leaving!\n");
}

void DataManager::set_num_blocks_per_burst(index_t num_blocks_per_burst)
{
	SG_SDEBUG("Entering!\n");
	REQUIRE(num_blocks_per_burst>0,
		   	"Number of blocks per burst (%d) has to be greater than 0!\n",
			num_blocks_per_burst);

	index_t blocksize=0;
	using fetcher_type=std::unique_ptr<DataFetcher>;
	std::for_each(fetchers.begin(), fetchers.end(), [&blocksize](fetcher_type& f)
	{
		blocksize+=f->m_block_details.m_blocksize;
	});
	REQUIRE(blocksize>0,
			"Blocksizes are not set!\n");

	index_t max_num_blocks_per_burst=get_num_samples()/blocksize;
	REQUIRE(num_blocks_per_burst<=max_num_blocks_per_burst,
			"There can only be %d blocks per burst given the blocksize (%d)!",
			max_num_blocks_per_burst, blocksize);

	for (size_t i=0; i<fetchers.size(); ++i)
		fetchers[i]->fetch_blockwise().with_num_blocks_per_burst(num_blocks_per_burst);
	SG_SDEBUG("Leaving!\n");
}

InitPerFeature DataManager::samples_at(size_t i)
{
	SG_SDEBUG("Entering!\n");
	REQUIRE(i<fetchers.size(),
			"Value of i (%d) should be between 0 and %d, inclusive!",
			i, fetchers.size()-1);
	SG_SDEBUG("Leaving!\n");
	return InitPerFeature(fetchers[i]);
}

CFeatures* DataManager::samples_at(size_t i) const
{
	SG_SDEBUG("Entering!\n");
	REQUIRE(i<fetchers.size(),
			"Value of i (%d) should be between 0 and %d, inclusive!",
			i, fetchers.size()-1);
	SG_SDEBUG("Leaving!\n");
	return fetchers[i]->m_samples.get();
}

index_t& DataManager::num_samples_at(size_t i)
{
	SG_SDEBUG("Entering!\n");
	REQUIRE(i<fetchers.size(),
			"Value of i (%d) should be between 0 and %d, inclusive!",
			i, fetchers.size()-1);
	SG_SDEBUG("Leaving!\n");
	return fetchers[i]->m_num_samples;
}

const index_t DataManager::num_samples_at(size_t i) const
{
	SG_SDEBUG("Entering!\n");
	REQUIRE(i<fetchers.size(),
			"Value of i (%d) should be between 0 and %d, inclusive!",
			i, fetchers.size()-1);
	SG_SDEBUG("Leaving!\n");
	return fetchers[i]->m_num_samples;
}

const index_t DataManager::blocksize_at(size_t i) const
{
	SG_SDEBUG("Entering!\n");
	REQUIRE(i<fetchers.size(),
			"Value of i (%d) should be between 0 and %d, inclusive!",
			i, fetchers.size()-1);
	SG_SDEBUG("Leaving!\n");
	return fetchers[i]->m_block_details.m_blocksize;
}

void DataManager::start()
{
	SG_SDEBUG("Entering!\n");
	using fetcher_type=std::unique_ptr<DataFetcher>;
	std::for_each(fetchers.begin(), fetchers.end(), [](fetcher_type& f) { f->start(); });
	SG_SDEBUG("Leaving!\n");
}

NextSamples DataManager::next()
{
	SG_SDEBUG("Entering!\n");
	NextSamples next_samples(fetchers.size());
	// fetch a number of blocks (per burst) from each distribution
	for (size_t i=0; i<fetchers.size(); ++i)
	{
		auto feats=fetchers[i]->next();
		if (feats!=nullptr)
		{
			auto blocksize=fetchers[i]->m_block_details.m_blocksize;
			auto num_blocks_curr_burst=feats->get_num_vectors()/blocksize;
			if (next_samples.m_num_blocks==0)
				next_samples.m_num_blocks=num_blocks_curr_burst;
			else
				ASSERT(next_samples.m_num_blocks==num_blocks_curr_burst);

			// next samples are gonna hold one feats obj per block for this burst
			next_samples[i].resize(num_blocks_curr_burst);
			SGVector<index_t> inds(blocksize);
			std::iota(inds.vector, inds.vector + inds.vlen, 0);
			for (auto j=0; j<num_blocks_curr_burst; ++j)
			{
				// subset each block and clone it separately
				feats->add_subset(inds);
				auto block=static_cast<CFeatures*>(feats->clone());
				next_samples[i][j]=std::shared_ptr<CFeatures>(block, [](CFeatures* ptr) { SG_UNREF(ptr); });
				feats->remove_subset();
				std::for_each(inds.vector, inds.vector+inds.vlen, [&blocksize](index_t& val) { val+=blocksize; });
			}
		}
	}
	SG_SDEBUG("Leaving!\n");
	return next_samples;
}

void DataManager::end()
{
	SG_SDEBUG("Entering!\n");
	using fetcher_type=std::unique_ptr<DataFetcher>;
	std::for_each(fetchers.begin(), fetchers.end(), [](fetcher_type& f) { f->end(); });
	SG_SDEBUG("Leaving!\n");
}

void DataManager::reset()
{
	SG_SDEBUG("Entering!\n");
	using fetcher_type=std::unique_ptr<DataFetcher>;
	std::for_each(fetchers.begin(), fetchers.end(), [](fetcher_type& f) { f->reset(); });
	SG_SDEBUG("Leaving!\n");
}
