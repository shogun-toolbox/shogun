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

DataFetcher::DataFetcher() : m_num_samples(0), m_samples(nullptr),
	train_test_subset_used(false)
{
}

DataFetcher::DataFetcher(CFeatures* samples) : m_samples(samples),
	train_test_subset_used(false)
{
	REQUIRE(m_samples!=nullptr, "Samples cannot be null!\n");
	SG_REF(m_samples);
	m_num_samples=m_samples->get_num_vectors();
	m_train_test_details.set_total_num_samples(m_num_samples);
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

void DataFetcher::set_train_test_ratio(float64_t train_test_ratio)
{
	m_num_samples=m_train_test_details.get_total_num_samples();
	REQUIRE(m_num_samples>0, "Number of samples is not set!\n");
	index_t num_training_samples=m_num_samples*train_test_ratio/(train_test_ratio+1);
	m_train_test_details.set_num_training_samples(num_training_samples);
	SG_SINFO("Must set the train/test mode by calling set_train_mode(True/False)!\n");
}

float64_t DataFetcher::get_train_test_ratio() const
{
	return float64_t(m_train_test_details.get_num_training_samples())/m_train_test_details.get_num_test_samples();
}

void DataFetcher::set_train_mode(bool train_mode)
{
	m_train_test_details.train_mode=train_mode;
	// TODO put the following in another methods
	index_t start_index=0;
	if (m_train_test_details.train_mode)
	{
		m_num_samples=m_train_test_details.get_num_training_samples();
		if (m_num_samples==0)
			SG_SERROR("The number of training samples is 0! Please set a valid train-test ratio\n");
		SG_SINFO("Using %d number of samples for training!\n", m_num_samples);
	}
	else
	{
		m_num_samples=m_train_test_details.get_num_test_samples();
		SG_SINFO("Using %d number of samples for testing!\n", m_num_samples);
		start_index=m_train_test_details.get_num_training_samples();
		if (start_index==0)
		{
			if (train_test_subset_used)
			{
				m_samples->remove_subset();
				train_test_subset_used=false;
			}
			return;
		}
	}
	SGVector<index_t> inds(m_num_samples);
	std::iota(inds.data(), inds.data()+inds.size(), start_index);
	if (train_test_subset_used)
		m_samples->remove_subset();
	m_samples->add_subset(inds);
	train_test_subset_used=true;
}

void DataFetcher::set_xvalidation_mode(bool xvalidation_mode)
{
//	using fetcher_type=std::unique_ptr<DataFetcher>;
//	std::for_each(fetchers.begin(), fetchers.end(), [&train_mode](fetcher_type& f)
//	{
//		f->set_xvalidation_mode(xvalidation_mode);
//	});
}

index_t DataFetcher::get_num_folds() const
{
	return 1+ceil(get_train_test_ratio());
}

void DataFetcher::use_fold(index_t idx)
{
	auto num_folds=get_num_folds();
	REQUIRE(idx>=0, "The index (%d) has to be between 0 and %d, both inclusive!\n", idx, num_folds-1);
	REQUIRE(idx<num_folds, "The index (%d) has to be between 0 and %d, both inclusive!\n", idx, num_folds-1);

	auto num_per_fold=m_train_test_details.get_total_num_samples()/num_folds;

	if (train_test_subset_used)
		m_samples->remove_subset();

	SGVector<index_t> inds;
	auto start_idx=idx*num_per_fold;
	auto num_samples=0;

	if (m_train_test_details.train_mode)
	{
		num_samples=m_train_test_details.get_num_training_samples();
		inds=SGVector<index_t>(num_samples);
		std::iota(inds.data(), inds.data()+inds.size(), 0);
		if (start_idx<inds.size())
		{
			std::for_each(inds.data()+start_idx, inds.data()+inds.size(), [&num_per_fold](index_t& val)
			{
				val+=num_per_fold;
			});
		}
	}
	else
	{
		num_samples=m_train_test_details.get_num_test_samples();
		inds=SGVector<index_t>(num_samples);
		std::iota(inds.data(), inds.data()+inds.size(), start_idx);
		m_samples->add_subset(inds);
	}
	inds.display_vector("inds");
	m_samples->add_subset(inds);
}

void DataFetcher::set_blockwise(bool blockwise)
{
	if (blockwise)
	{
		m_block_details=last_blockwise_details;
		SG_SDEBUG("Restoring the blockwise details!\n");
		m_block_details.m_full_data=false;
	}
	else
	{
		last_blockwise_details=m_block_details;
		SG_SDEBUG("Saving the blockwise details!\n");
		m_block_details=BlockwiseDetails();
	}
}

void DataFetcher::start()
{
	REQUIRE(m_num_samples>0, "Number of samples is 0!\n");
	if (m_block_details.m_full_data || m_block_details.m_blocksize>m_num_samples)
	{
		SG_SINFO("Fetching entire data (%d samples)!\n", m_num_samples);
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
	m_block_details.m_full_data=false;
	return m_block_details;
}
