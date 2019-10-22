/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Thoralf Klein, Bjoern Esser, Sergey Lisitsyn
 */

#include <shogun/lib/IndexBlock.h>

using namespace shogun;

IndexBlock::IndexBlock() : SGObject(),
	m_min_index(0), m_max_index(0),
	m_weight(1.0)
{
}

IndexBlock::IndexBlock(index_t min_index, index_t max_index,
             float64_t weight, const char* name) :
	SGObject(), m_min_index(min_index), m_max_index(max_index),
	m_weight(weight)
{
}

IndexBlock::~IndexBlock()
{

}

void IndexBlock::add_sub_block(std::shared_ptr<IndexBlock> sub_block)
{
	ASSERT(sub_block->get_min_index()>=m_min_index)
	ASSERT(sub_block->get_max_index()<=m_max_index)
	m_sub_blocks.push_back(sub_block);
}

std::vector<std::shared_ptr<IndexBlock>> IndexBlock::get_sub_blocks()
{
	return m_sub_blocks;
}

int32_t IndexBlock::get_num_sub_blocks()
{
	return m_sub_blocks.size();
}
