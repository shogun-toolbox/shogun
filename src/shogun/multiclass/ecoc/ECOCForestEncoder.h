/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#ifndef ECOCFORESTENCODER_H__
#define ECOCFORESTENCODER_H__

#include <multiclass/ecoc/ECOCDiscriminantEncoder.h>

namespace shogun
{

/** Forest ECOC Encoder.
 *
 * A data-dependent ECOC coding scheme that learns a tree-style codebook. See the
 * following paper for details
 *
 *   Sergio Escalera, Oriol Pujol, Petia Radeva. Boosted Landmarks of
 *   Contextual Descriptors and Forest-ECOC: A novel framework to detect and
 *   classify objects in cluttered scenes. Pattern Recognition Letters, 2007.
 *
 */
class CECOCForestEncoder: public CECOCDiscriminantEncoder
{
public:
    /** constructor */
    CECOCForestEncoder();

    /** destructor */
    virtual ~CECOCForestEncoder() {}

    /** get name */
    virtual const char* get_name() const { return "ECOCForestEncoder"; }

    /** get number of trees */
    int32_t get_num_trees() const { return m_num_trees; }

    /** set number of trees */
    void set_num_trees(int32_t num_trees);
};

} /*  shogun */

#endif /* end of include guard: ECOCFORESTENCODER_H__ */

