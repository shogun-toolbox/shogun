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

#include <shogun/lib/common.h>

#ifndef BLOCK_WISE_DETAILS_H__
#define BLOCK_WISE_DETAILS_H__

namespace shogun
{

namespace internal
{

class BlockwiseDetails
{
	friend class DataFetcher;
	friend class StreamingDataFetcher;
	friend class DataManager;
public:
	BlockwiseDetails();
	BlockwiseDetails& with_blocksize(index_t blocksize);
	BlockwiseDetails& with_num_blocks_per_burst(index_t num_blocks_per_burst);
private:
	index_t m_blocksize;
	index_t m_num_blocks_per_burst;
	index_t m_max_num_samples_per_burst;
	// the following will be set by data fetchers
	index_t m_next_block_index;
	index_t m_total_num_blocks;
};

}

}
#endif // BLOCK_WISE_DETAILS_H__
