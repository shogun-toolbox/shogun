/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include "features/Subset.h"
#include "lib/io.h"

using namespace shogun;

Subset::Subset()
{
	m_subset_idx=NULL, m_subset_len=0;
}

Subset::~Subset()
{
	delete[] m_subset_idx;
}

void Subset::set_subset(index_t subset_len, index_t* subset_idx)
{
	delete[] m_subset_idx;

	m_subset_idx=subset_idx;
	m_subset_len=subset_len;
}

void Subset::set_subset(index_t* subset_idx, index_t subset_len)
{
	ASSERT(subset_idx);

	delete[] m_subset_idx;
	m_subset_idx=NULL;

	size_t length=sizeof(index_t)*subset_len;

	m_subset_idx=(index_t*) SG_MALLOC(length);

	memcpy(m_subset_idx, subset_idx, length);
}

void Subset::remove_subset()
{
	delete[] m_subset_idx;
	m_subset_idx=NULL;
	m_subset_len=0;
}

void Subset::get_subset(index_t** subset_idx, index_t* subset_len)
{
	if (!m_subset_idx)
		SG_SERROR("no subset set to copy!\n");

	int64_t length=sizeof(index_t)*m_subset_len;

	*subset_len=m_subset_len;
	*subset_idx=(index_t*) SG_MALLOC(length);
	memcpy(*subset_idx, m_subset_idx, length);
}
