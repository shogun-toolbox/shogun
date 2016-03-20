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

#include <shogun/statistics/experimental/internals/BlockwiseDetails.h>

using namespace shogun;
using namespace internal;

BlockwiseDetails::BlockwiseDetails()
: m_blocksize(0), m_num_blocks_per_burst(1), m_max_num_samples_per_burst(0),
  m_next_block_index(0), m_total_num_blocks(0)
{
}

BlockwiseDetails& BlockwiseDetails::with_blocksize(index_t blocksize)
{
	m_blocksize = blocksize;
	m_max_num_samples_per_burst = m_blocksize * m_num_blocks_per_burst;
	return *this;
}

BlockwiseDetails& BlockwiseDetails::with_num_blocks_per_burst(index_t num_blocks_per_burst)
{
	m_num_blocks_per_burst = num_blocks_per_burst;
	m_max_num_samples_per_burst = m_blocksize * m_num_blocks_per_burst;
	return *this;
}
