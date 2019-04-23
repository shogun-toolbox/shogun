/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Thoralf Klein, Soeren Sonnenburg, Bjoern Esser, Sergey Lisitsyn
 */

#include <shogun/lib/IndexBlockRelation.h>
#include <shogun/lib/IndexBlock.h>
#include <shogun/lib/List.h>

using namespace shogun;

bool IndexBlockRelation::check_blocks_list(std::shared_ptr<List> blocks)
{
	int32_t n_sub_blocks = blocks->get_num_elements();
	index_t* min_idxs = SG_MALLOC(index_t, n_sub_blocks);
	index_t* max_idxs = SG_MALLOC(index_t, n_sub_blocks);
	index_t* block_idxs_min = SG_MALLOC(index_t, n_sub_blocks);
	index_t* block_idxs_max = SG_MALLOC(index_t, n_sub_blocks);
	auto iter_block = std::static_pointer_cast<IndexBlock>(blocks->get_first_element());
	for (int32_t i=0; i<n_sub_blocks; i++)
	{
		min_idxs[i] = iter_block->get_min_index();
		max_idxs[i] = iter_block->get_max_index();
		block_idxs_min[i] = i;
		block_idxs_max[i] = i;
		iter_block = std::static_pointer_cast<IndexBlock>(blocks->get_next_element());
	}
	Math::qsort_index(min_idxs, block_idxs_min, n_sub_blocks);
	Math::qsort_index(max_idxs, block_idxs_max, n_sub_blocks);

	for (int32_t i=0; i<n_sub_blocks; i++)
	{
		if (block_idxs_min[i] != block_idxs_max[i])
			error("Blocks do overlap and it is not supported");
	}
	if (min_idxs[0] != 0)
		error("Block with smallest indices start from {} while 0 is required", min_idxs[0]);

	for (int32_t i=1; i<n_sub_blocks; i++)
	{
		if (min_idxs[i] > max_idxs[i-1])
			error("There is an unsupported gap between {} and {} vectors", max_idxs[i-1], min_idxs[i]);
		else if (min_idxs[i] < max_idxs[i-1])
			error("Blocks do overlap and it is not supported");
	}

	SG_FREE(min_idxs);
	SG_FREE(max_idxs);
	SG_FREE(block_idxs_min);
	SG_FREE(block_idxs_max);
	return true;
}
