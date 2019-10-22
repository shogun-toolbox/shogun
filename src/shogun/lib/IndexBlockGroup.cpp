/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Thoralf Klein, Soeren Sonnenburg
 */

#include <shogun/lib/IndexBlockGroup.h>
#include <shogun/lib/IndexBlock.h>

using namespace shogun;

IndexBlockGroup::IndexBlockGroup() : IndexBlockRelation()
{
}

IndexBlockGroup::~IndexBlockGroup()
{

}

void IndexBlockGroup::add_block(std::shared_ptr<IndexBlock> block)
{
	m_blocks.push_back(block);
}

void IndexBlockGroup::remove_block(std::shared_ptr<IndexBlock> block)
{
	not_implemented(SOURCE_LOCATION);
}

SGVector<index_t> IndexBlockGroup::get_SLEP_ind()
{
	check_blocks_list(m_blocks);
	int32_t n_sub_blocks = m_blocks.size();
	SG_DEBUG("Number of sub-blocks = {}", n_sub_blocks)
	
	SGVector<index_t> ind(n_sub_blocks+1);
	ind[0] = 0;
	int32_t i = 0;
	for (const auto& it: m_blocks)
	{
		ind[i+1] = it->get_max_index();
		i++;
	}

	return ind;
}
