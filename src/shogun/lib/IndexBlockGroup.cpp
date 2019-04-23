/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Thoralf Klein, Soeren Sonnenburg
 */

#include <shogun/lib/IndexBlockGroup.h>
#include <shogun/lib/IndexBlock.h>
#include <shogun/lib/List.h>

using namespace shogun;

IndexBlockGroup::IndexBlockGroup() : IndexBlockRelation()
{
	m_blocks = std::make_shared<List>(true);
}

IndexBlockGroup::~IndexBlockGroup()
{

}

void IndexBlockGroup::add_block(std::shared_ptr<IndexBlock> block)
{
	m_blocks->push(block);
}

void IndexBlockGroup::remove_block(std::shared_ptr<IndexBlock> block)
{
	not_implemented(SOURCE_LOCATION);
}

SGVector<index_t> IndexBlockGroup::get_SLEP_ind()
{
	check_blocks_list(m_blocks);
	int32_t n_sub_blocks = m_blocks->get_num_elements();
	SG_DEBUG("Number of sub-blocks = {}", n_sub_blocks)
	SGVector<index_t> ind(n_sub_blocks+1);

	auto iterator = std::dynamic_pointer_cast<IndexBlock>(m_blocks->get_first_element());
	ind[0] = 0;
	int32_t i = 0;
	do
	{
		ind[i+1] = iterator->get_max_index();
		i++;
	}
	while ((iterator = std::dynamic_pointer_cast<IndexBlock>(m_blocks->get_next_element())) != NULL);
	//ind.display_vector("ind");

	return ind;
}
