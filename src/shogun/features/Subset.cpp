/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Heiko Strathmann, Evgeniy Andreev, Soumyajit De
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
