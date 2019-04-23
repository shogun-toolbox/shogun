/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Heiko Strathmann, Evgeniy Andreev, Soumyajit De
 */

#include <shogun/features/Subset.h>
#include <shogun/base/Parameter.h>

using namespace shogun;

Subset::Subset()
{
	init();
}

Subset::Subset(const SGVector<index_t>& subset_idx)
{
	init();

	m_subset_idx = subset_idx.clone();
}

Subset::~Subset()
{
}

void Subset::init()
{
	SG_ADD(&m_subset_idx, "subset", "Vector of subset indices");
}
