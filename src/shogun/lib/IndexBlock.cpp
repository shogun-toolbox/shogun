/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#include <shogun/lib/IndexBlock.h>

using namespace shogun;

CIndexBlock::CIndexBlock() : CSGObject(),
	m_min_index(0), m_max_index(0),
	m_weight(1.0), m_sub_blocks(NULL)
{
	m_sub_blocks = new CList(true);
	SG_REF(m_sub_blocks);
}

CIndexBlock::CIndexBlock(index_t min_index, index_t max_index,
             float64_t weight, const char* name) :
	CSGObject(), m_min_index(min_index), m_max_index(max_index),
	m_weight(weight), m_sub_blocks(NULL)
{
	m_sub_blocks = new CList(true);
	SG_REF(m_sub_blocks);
}

CIndexBlock::~CIndexBlock()
{
	SG_UNREF(m_sub_blocks);
}

void CIndexBlock::add_sub_block(CIndexBlock* sub_block)
{
	ASSERT(sub_block->get_min_index()>=m_min_index)
	ASSERT(sub_block->get_max_index()<=m_max_index)
	m_sub_blocks->append_element(sub_block);
}

CList* CIndexBlock::get_sub_blocks()
{
	SG_REF(m_sub_blocks);
	return m_sub_blocks;
}

int32_t CIndexBlock::get_num_sub_blocks()
{
	return m_sub_blocks->get_num_elements();
}
