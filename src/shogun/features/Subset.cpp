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

CSubset::CSubset(SGVector<index_t> subset_idx)
{
	init();

	m_subset_idx=subset_idx;
}

CSubset::~CSubset()
{
}

void CSubset::init()
{
	SG_ADD(&m_subset_idx, "subset", "Vector of subset indices",
			MS_NOT_AVAILABLE);
}
