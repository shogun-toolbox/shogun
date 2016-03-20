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

#include <iostream> // TODO remove
#include <shogun/statistics/experimental/internals/DataManager.h>
#include <shogun/statistics/experimental/internals/NextSamples.h>
#include <shogun/statistics/experimental/internals/DataFetcher.h>
#include <shogun/statistics/experimental/internals/DataFetcherFactory.h>
#include <shogun/features/Features.h>

using namespace shogun;
using namespace internal;

DataManager::DataManager(index_t num_distributions)
{
	fetchers.resize(num_distributions);
	std::fill(fetchers.begin(), fetchers.end(), nullptr);
}

DataManager::~DataManager()
{
}

index_t DataManager::get_num_samples() const
{
	index_t n = 0;
	using fetcher_type = const std::unique_ptr<DataFetcher>;
	if (std::any_of(fetchers.begin(), fetchers.end(), [](fetcher_type& f) { return f->m_num_samples == 0; }))
	{
		std::cout << "number of samples from all the distributions are not set" << std::endl;
	}
	else
	{
		std::for_each(fetchers.begin(), fetchers.end(), [&n](fetcher_type& f) { n+= f->m_num_samples; });
	}
	return n;
}

index_t DataManager::get_min_blocksize() const
{
	index_t min_blocksize = 0;
	using fetcher_type = const std::unique_ptr<DataFetcher>;
	if (std::any_of(fetchers.begin(), fetchers.end(), [](fetcher_type& f) { return f->m_num_samples == 0; }))
	{
		std::cout << "number of samples from all the distributions are not set" << std::endl;
	}
	else
	{
		index_t divisor = 0;
		std::function<index_t(index_t, index_t)> gcd = [&gcd](index_t m, index_t n)
		{
			return n == 0 ? m : gcd(n, m % n);
		};
		for (auto i = 0; i < fetchers.size(); ++i)
		{
			divisor = gcd(divisor, fetchers[i]->m_num_samples);
		}
		min_blocksize = get_num_samples() / divisor;
	}
	std::cout << "min blocksize is " << min_blocksize << std::endl;
	return min_blocksize;
}

void DataManager::set_blocksize(index_t blocksize)
{
	auto n = get_num_samples();

	ASSERT(n > 0);
	ASSERT(blocksize > 0 && blocksize <= n);
	ASSERT(n % blocksize == 0);

	for (auto i = 0; i < fetchers.size(); ++i)
	{
		index_t m = fetchers[i]->m_num_samples;
		ASSERT((blocksize * m) % n == 0);
		fetchers[i]->fetch_blockwise().with_blocksize(blocksize * m / n);
		std::cout << "block[" << i << "].size = " << blocksize * m / n << std::endl;
	}
}

void DataManager::set_num_blocks_per_burst(index_t num_blocks_per_burst)
{
	ASSERT(num_blocks_per_burst > 0);

	index_t blocksize = 0;
	using fetcher_type = std::unique_ptr<DataFetcher>;
	std::for_each(fetchers.begin(), fetchers.end(), [&blocksize](fetcher_type& f)
	{
		blocksize += f->m_block_details.m_blocksize;
	});
	ASSERT(blocksize > 0);

	index_t max_num_blocks_per_burst = get_num_samples() / blocksize;
	ASSERT(num_blocks_per_burst <= max_num_blocks_per_burst);

	for (auto i = 0; i < fetchers.size(); ++i)
	{
		fetchers[i]->fetch_blockwise().with_num_blocks_per_burst(num_blocks_per_burst);
	}
}

InitPerFeature DataManager::samples_at(index_t i)
{
	std::cout << "DataManager::samples_at()" << std::endl;
	ASSERT(i < fetchers.size());
	return InitPerFeature(fetchers[i]);
}

CFeatures* DataManager::samples_at(index_t i) const
{
	std::cout << "DataManager::samples_at() const" << std::endl;
	ASSERT(i < fetchers.size());
	return fetchers[i]->m_samples.get();
}

index_t& DataManager::num_samples_at(index_t i)
{
	std::cout << "DataManager::num_samples_at()" << std::endl;
	ASSERT(i < fetchers.size());
	return fetchers[i]->m_num_samples;
}

const index_t DataManager::num_samples_at(index_t i) const
{
	std::cout << "DataManager::num_samples_at() const" << std::endl;
	ASSERT(i < fetchers.size());
	return fetchers[i]->m_num_samples;
}

const index_t DataManager::blocksize_at(index_t i) const
{
	std::cout << "DataManager::blocksize_at() const" << std::endl;
	ASSERT(i < fetchers.size());
	return fetchers[i]->m_block_details.m_blocksize;
}

void DataManager::start()
{
	using fetcher_type = std::unique_ptr<DataFetcher>;
	std::for_each(fetchers.begin(), fetchers.end(), [](fetcher_type& f) { f->start(); });
}

NextSamples DataManager::next()
{
	std::cout << "DataManager::next()" << std::endl;
	NextSamples next_samples(fetchers.size());
	// fetch a number of blocks (per burst) from each distribution
	for (auto i = 0; i < fetchers.size(); ++i)
	{
		auto feats = fetchers[i]->next();
		if (feats != nullptr)
		{
			auto blocksize = fetchers[i]->m_block_details.m_blocksize;
			auto num_blocks_curr_burst = feats->get_num_vectors() / blocksize;
			if (next_samples.m_num_blocks == 0)
			{
				next_samples.m_num_blocks = num_blocks_curr_burst;
			}
			else
			{
				ASSERT(next_samples.m_num_blocks == num_blocks_curr_burst);
			}

			// next samples are gonna hold one feats obj per block for this burst
			next_samples[i].resize(num_blocks_curr_burst);
			SGVector<index_t> inds(blocksize);
			std::iota(inds.vector, inds.vector + inds.vlen, 0);
			for (auto j = 0; j < num_blocks_curr_burst; ++j)
			{
				// subset each block and clone it separately
				feats->add_subset(inds);
				auto block = static_cast<CFeatures*>(feats->clone());
				next_samples[i][j] = std::shared_ptr<CFeatures>(block, [](CFeatures* ptr) { SG_UNREF(ptr); });
				feats->remove_subset();
				std::for_each(inds.vector, inds.vector + inds.vlen, [&blocksize](index_t& val) { val += blocksize; });
			}
		}
	}
	return next_samples;
}

void DataManager::end()
{
	using fetcher_type = std::unique_ptr<DataFetcher>;
	std::for_each(fetchers.begin(), fetchers.end(), [](fetcher_type& f) { f->end(); });
}

void DataManager::reset()
{
	using fetcher_type = std::unique_ptr<DataFetcher>;
	std::for_each(fetchers.begin(), fetchers.end(), [](fetcher_type& f) { f->reset(); });
}
