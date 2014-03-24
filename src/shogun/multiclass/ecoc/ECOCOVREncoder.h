/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#ifndef ECOCOVRENCODER_H__
#define ECOCOVRENCODER_H__

#include <shogun/lib/config.h>
#include <shogun/multiclass/ecoc/ECOCEncoder.h>

namespace shogun
{

/** One-vs-Rest Encoder */
class CECOCOVREncoder: public CECOCEncoder
{
public:
    /** constructor */
    CECOCOVREncoder() {}

    /** destructor */
    virtual ~CECOCOVREncoder() {}

    /** get name */
    virtual const char* get_name() const
    {
        return "ECOCOVREncoder";
    }

    /** init codebook.
     * @param num_classes number of classes in this problem
     */
    virtual SGMatrix<int32_t> create_codebook(int32_t num_classes);
};

}

#endif /* end of include guard: ECOCOVRENCODER_H__ */
