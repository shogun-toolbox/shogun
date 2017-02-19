/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2016 Soumyajit De
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

#include <shogun/lib/common.h>

#ifndef BLOCK_WISE_DETAILS_H__
#define BLOCK_WISE_DETAILS_H__

namespace shogun
{

namespace internal
{

/**
 * @brief Class that holds block-details for the data-fetchers.
 * There are one instance of this class per fetcher.
 */
class BlockwiseDetails
{
	friend class DataFetcher;
	friend class StreamingDataFetcher;
	friend class DataManager;

public:

	/**
	 * Default constructor.
	 */
	BlockwiseDetails();

	/**
	 * Method that sets the blocksize for current fetcher.
	 * @param blocksize the size of the block
	 * @return an instance of the current object
	 */
	BlockwiseDetails& with_blocksize(index_t blocksize);

	/**
	 * Method that sets the number of blocks to be fetched per burst for current fetcher.
	 * @param num_blocks_per_burst the number of blocks to be fetched per burst
	 * @return an instance of the current object
	 */
	BlockwiseDetails& with_num_blocks_per_burst(index_t num_blocks_per_burst);

private:

	/** The size of the blocks */
	index_t m_blocksize;

	/** The number of blocks fetched per burst */
	index_t m_num_blocks_per_burst;

	/** The maximum number of samples fetched per burst */
	index_t m_max_num_samples_per_burst;

	/** Index for the next block to be fetched. Set by data fetchers */
	index_t m_next_block_index;

	/** Total number of blocks to be fetched. Set by data fetchers */
	index_t m_total_num_blocks;

	/** Whether the block should consist of full data (i.e. no block at all) */
	bool m_full_data;
};

}

}
#endif // BLOCK_WISE_DETAILS_H__
