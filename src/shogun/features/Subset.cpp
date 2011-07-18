/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/features/Subset.h>
#include <shogun/io/io.h>
#include <shogun/base/Parameter.h>

using namespace shogun;

CSubset::CSubset() : m_subset_idx(SGVector<index_t>())
{
	init();
}

CSubset::CSubset(SGVector<index_t> subset_idx) : m_subset_idx(subset_idx)
{
	init();
}

CSubset::~CSubset() {
	delete[] m_subset_idx.vector;
}

CSubset* CSubset::duplicate() {
	SGVector<index_t> idx_copy(m_subset_idx.vlen);

	memcpy(idx_copy.vector, m_subset_idx.vector,
			sizeof(index_t)*m_subset_idx.vlen);

	CSubset* copy_subset=new CSubset(idx_copy);

	return copy_subset;
}

void CSubset::init() {
	m_parameters->add((CSGObject**)&m_subset_idx, "subset", "Vector of subset indices");
}
