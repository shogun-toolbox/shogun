/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Chiyuan Zhang, Soeren Sonnenburg
 */

#include <shogun/multiclass/ecoc/ECOCForestEncoder.h>

using namespace shogun;

ECOCForestEncoder::ECOCForestEncoder()
{
    m_num_trees = 3;
    SG_ADD(&m_num_trees, "num_trees", "number of trees");
}

void ECOCForestEncoder::set_num_trees(int32_t num_trees)
{
    if (num_trees < 1)
        error("number of trees ({}) should be >= 1", num_trees);
    m_num_trees = num_trees;
}
