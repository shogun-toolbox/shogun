/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#include <shogun/lib/IndexBlockGroup.h>
#include <shogun/lib/IndexBlock.h>
#include <shogun/lib/List.h>

using namespace shogun;

CIndexBlockGroup::CIndexBlockGroup() : CIndexBlockRelation()
{
	m_blocks = new CList(true);
}

CIndexBlockGroup::~CIndexBlockGroup()
{
	SG_UNREF(m_blocks);
}

void CIndexBlockGroup::add_block(CIndexBlock* block)
{
	m_blocks->push(block);
}

void CIndexBlockGroup::remove_block(CIndexBlock* block)
{
	SG_NOTIMPLEMENTED
}

SGVector<index_t> CIndexBlockGroup::get_SLEP_ind()
{
	check_blocks_list(m_blocks);
	int32_t n_sub_blocks = m_blocks->get_num_elements();
	SG_DEBUG("Number of sub-blocks = %d\n", n_sub_blocks)
	SGVector<index_t> ind(n_sub_blocks+1);

	CIndexBlock* iterator = (CIndexBlock*)(m_blocks->get_first_element());
	ind[0] = 0;
	int32_t i = 0;
	do
	{
		ind[i+1] = iterator->get_max_index();
		SG_UNREF(iterator);
		i++;
	}
	while ((iterator = (CIndexBlock*)m_blocks->get_next_element()) != NULL);
	//ind.display_vector("ind");

	return ind;
}
