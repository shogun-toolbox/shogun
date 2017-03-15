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
#include <shogun/features/Features.h>
#include <shogun/statistical_testing/internals/DataFetcher.h>

using namespace shogun;
using namespace internal;

DataFetcher::DataFetcher() : m_num_samples(0), train_test_mode(false),
	train_mode(false), m_samples(nullptr), features_shuffled(false)
{
}

DataFetcher::DataFetcher(CFeatures* samples) : train_test_mode(false),
   	train_mode(false), m_samples(samples), features_shuffled(false)
{
	REQUIRE(m_samples!=nullptr, "Samples cannot be null!\n");
	SG_REF(m_samples);
	m_num_samples=m_samples->get_num_vectors();
}

DataFetcher::~DataFetcher()
{
	SG_UNREF(m_samples);
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

void DataFetcher::set_train_test_mode(bool on)
{
	train_test_mode=on;
}

bool DataFetcher::is_train_test_mode() const
{
	return train_test_mode;
}

void DataFetcher::set_train_mode(bool on)
{
	train_mode=on;
}

bool DataFetcher::is_train_mode() const
{
	return train_mode;
}

void DataFetcher::set_train_test_ratio(float64_t ratio)
{
	train_test_ratio=ratio;
}

float64_t DataFetcher::get_train_test_ratio() const
{
	return train_test_ratio;
}

void DataFetcher::shuffle_features()
{
	REQUIRE(train_test_mode, "This method is allowed only when Train/Test method is active!\n");
	if (features_shuffled)
	{
		SG_SWARNING("Features are already shuffled! Call to shuffle_features() has no effect."
		"If you want to reshuffle, please call unshuffle_features() first and then call this method!\n");
	}
	else
	{
		const index_t size=m_samples->get_num_vectors();
		SG_SDEBUG("Current number of feature vectors = %d\n", size);
		if (shuffle_subset.size()<size)
		{
			SG_SDEBUG("Resizing the shuffle indices vector (from %d to %d)\n", shuffle_subset.size(), size);
			shuffle_subset=SGVector<index_t>(size);
		}
		std::iota(shuffle_subset.data(), shuffle_subset.data()+shuffle_subset.size(), 0);
		CMath::permute(shuffle_subset);
//		shuffle_subset.display_vector("shuffle_subset");

		SG_SDEBUG("Shuffling %d feature vectors\n", size);
		m_samples->add_subset(shuffle_subset);

		features_shuffled=true;
	}
}

void DataFetcher::unshuffle_features()
{
	REQUIRE(train_test_mode, "This method is allowed only when Train/Test method is active!\n");
	if (features_shuffled)
	{
		m_samples->remove_subset();
		features_shuffled=false;
	}
	else
	{
		SG_SWARNING("Features are NOT shuffled! Call to unshuffle_features() has no effect."
		"If you want to reshuffle, please call shuffle_features() instead!\n");
	}
}

void DataFetcher::use_fold(index_t idx)
{
	allocate_active_subset();
	auto num_samples_per_fold=get_num_samples()/get_num_folds();
	auto start_idx=idx*num_samples_per_fold;
	if (train_mode)
	{
		std::iota(active_subset.data(), active_subset.data()+active_subset.size(), 0);
		if (start_idx<active_subset.size())
		{
			std::for_each(active_subset.data()+start_idx, active_subset.data()+active_subset.size(),
			[&num_samples_per_fold](index_t& val)
			{
				val+=num_samples_per_fold;
			});
		}
	}
	else
		std::iota(active_subset.data(), active_subset.data()+active_subset.size(), start_idx);
//	active_subset.display_vector("active_subset");
}

void DataFetcher::init_active_subset()
{
	allocate_active_subset();
	index_t start_index=0;
	if (!train_mode)
		start_index=m_samples->get_num_vectors()*train_test_ratio/(train_test_ratio+1);
	std::iota(active_subset.data(), active_subset.data()+active_subset.size(), start_index);
//	active_subset.display_vector("active_subset");
}

void DataFetcher::start()
{
	REQUIRE(get_num_samples()>0, "Number of samples is 0!\n");
	if (train_test_mode)
	{
		m_samples->add_subset(active_subset);
		SG_SDEBUG("Added active subset!\n");
		SG_SINFO("Currently active number of samples is %d\n", get_num_samples());
	}

	if (m_block_details.m_full_data || m_block_details.m_blocksize>get_num_samples())
	{
		SG_SINFO("Fetching entire data (%d samples)!\n", get_num_samples());
		m_block_details.with_blocksize(get_num_samples());
	}
	m_block_details.m_total_num_blocks=get_num_samples()/m_block_details.m_blocksize;
	reset();
}

CFeatures* DataFetcher::next()
{
	CFeatures* next_samples=nullptr;
	// figure out how many samples to fetch in this burst
	auto num_already_fetched=m_block_details.m_next_block_index*m_block_details.m_blocksize;
	auto num_more_samples=get_num_samples()-num_already_fetched;
	if (num_more_samples>0)
	{
		// create a shallow copy and add proper index subset
		next_samples=m_samples->shallow_subset_copy();
		auto num_samples_this_burst=std::min(m_block_details.m_max_num_samples_per_burst, num_more_samples);
		if (num_samples_this_burst<next_samples->get_num_vectors())
		{
			SGVector<index_t> inds(num_samples_this_burst);
			std::iota(inds.vector, inds.vector+inds.vlen, num_already_fetched);
			next_samples->add_subset(inds);
		}
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
	if (train_test_mode)
	{
		m_samples->remove_subset();
		SG_SDEBUG("Removed active subset!\n");
		SG_SINFO("Currently active number of samples is %d\n", get_num_samples());
	}
}

index_t DataFetcher::get_num_samples() const
{
	if (train_test_mode)
	{
		if (train_mode)
			return m_num_samples*train_test_ratio/(train_test_ratio+1);
		else
			return m_num_samples/(train_test_ratio+1);
	}
	return m_samples->get_num_vectors();
}

index_t DataFetcher::get_num_folds() const
{
	return 1+ceil(get_train_test_ratio());
}

index_t DataFetcher::get_num_training_samples() const
{
	return get_num_samples()*get_train_test_ratio()/(get_train_test_ratio()+1);
}

index_t DataFetcher::get_num_testing_samples() const
{
	return get_num_samples()/(get_train_test_ratio()+1);
}

BlockwiseDetails& DataFetcher::fetch_blockwise()
{
	m_block_details.m_full_data=false;
	return m_block_details;
}

void DataFetcher::allocate_active_subset()
{
	REQUIRE(train_test_mode, "This method is allowed only when Train/Test method is active!\n");
	index_t num_active_samples=0;
	if (train_mode)
	{
		num_active_samples=m_samples->get_num_vectors()*train_test_ratio/(train_test_ratio+1);
		SG_SINFO("Using %d number of samples for this fold as training samples!\n", num_active_samples);
	}
	else
	{
		num_active_samples=m_samples->get_num_vectors()/(train_test_ratio+1);
		SG_SINFO("Using %d number of samples for this fold as testing samples!\n", num_active_samples);
	}

	ASSERT(num_active_samples>0);
	if (active_subset.size()!=num_active_samples)
	{
		SG_SDEBUG("Resizing the active subset from %d to %d\n", active_subset.size(), num_active_samples);
		active_subset=SGVector<index_t>(num_active_samples);
	}
}
