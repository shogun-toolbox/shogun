/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#include <lib/IndexBlockRelation.h>
#include <lib/IndexBlock.h>
#include <lib/SGVector.h>

using namespace shogun;

bool CIndexBlockRelation::check_blocks_list(CList* blocks)
{
	int32_t n_sub_blocks = blocks->get_num_elements();
	index_t* min_idxs = SG_MALLOC(index_t, n_sub_blocks);
	index_t* max_idxs = SG_MALLOC(index_t, n_sub_blocks);
	index_t* block_idxs_min = SG_MALLOC(index_t, n_sub_blocks);
	index_t* block_idxs_max = SG_MALLOC(index_t, n_sub_blocks);
	CIndexBlock* iter_block = (CIndexBlock*)(blocks->get_first_element());
	for (int32_t i=0; i<n_sub_blocks; i++)
	{
		min_idxs[i] = iter_block->get_min_index();
		max_idxs[i] = iter_block->get_max_index();
		block_idxs_min[i] = i;
		block_idxs_max[i] = i;
		SG_UNREF(iter_block);
		iter_block = (CIndexBlock*)(blocks->get_next_element());
	}
	CMath::qsort_index(min_idxs, block_idxs_min, n_sub_blocks);
	CMath::qsort_index(max_idxs, block_idxs_max, n_sub_blocks);

	for (int32_t i=0; i<n_sub_blocks; i++)
	{
		if (block_idxs_min[i] != block_idxs_max[i])
			SG_ERROR("Blocks do overlap and it is not supported\n")
	}
	if (min_idxs[0] != 0)
		SG_ERROR("Block with smallest indices start from %d while 0 is required\n", min_idxs[0])

	for (int32_t i=1; i<n_sub_blocks; i++)
	{
		if (min_idxs[i] > max_idxs[i-1])
			SG_ERROR("There is an unsupported gap between %d and %d vectors\n", max_idxs[i-1], min_idxs[i])
		else if (min_idxs[i] < max_idxs[i-1])
			SG_ERROR("Blocks do overlap and it is not supported\n")
	}

	SG_FREE(min_idxs);
	SG_FREE(max_idxs);
	SG_FREE(block_idxs_min);
	SG_FREE(block_idxs_max);
	return true;
}
