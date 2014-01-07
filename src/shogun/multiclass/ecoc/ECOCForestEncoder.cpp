/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#include <multiclass/ecoc/ECOCForestEncoder.h>

using namespace shogun;

CECOCForestEncoder::CECOCForestEncoder()
{
    m_num_trees = 3;
    SG_ADD(&m_num_trees, "num_trees", "number of trees", MS_NOT_AVAILABLE);
}

void CECOCForestEncoder::set_num_trees(int32_t num_trees)
{
    if (num_trees < 1)
        SG_ERROR("number of trees (%d) should be >= 1", num_trees)
    m_num_trees = num_trees;
}
