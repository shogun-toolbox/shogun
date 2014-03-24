/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#ifndef ECOCOVOENCODER_H__
#define ECOCOVOENCODER_H__

#include <shogun/lib/config.h>
#include <shogun/multiclass/ecoc/ECOCEncoder.h>

namespace shogun
{

/** One-vs-One Encoder */
class CECOCOVOEncoder: public CECOCEncoder
{
public:
    /** constructor */
    CECOCOVOEncoder() {}

    /** destructor */
    virtual ~CECOCOVOEncoder() {}

    /** get name */
    virtual const char* get_name() const
    {
        return "ECOCOVOEncoder";
    }

    /** init codebook.
     * @param num_classes number of classes in this problem
     */
    virtual SGMatrix<int32_t> create_codebook(int32_t num_classes);
};

}

#endif /* end of include guard: ECOCOVOENCODER_H__ */
