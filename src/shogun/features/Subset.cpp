/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011-2012 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/features/Subset.h>
#include <shogun/base/Parameter.h>

using namespace shogun;

CSubset::CSubset()
{
	init();
}

CSubset::CSubset(const SGVector<index_t>& subset_idx)
{
	init();

	/* copy indices. TODO this is not needed once there is ref-counting for
	 * SGVectors */
	m_subset_idx=SGVector<index_t>(subset_idx.vlen);
	memcpy(m_subset_idx.vector, subset_idx.vector,
			subset_idx.vlen*sizeof(index_t));
}

CSubset::~CSubset() {
	/* TODO, change to UNREF, once it is possible */
	m_subset_idx.destroy_vector();
}

void CSubset::init() {
	SG_ADD((SGVector<index_t>*)&m_subset_idx, "subset",
			"Vector of subset indices", MS_NOT_AVAILABLE);

	m_subset_idx=SGVector<index_t>();
}
