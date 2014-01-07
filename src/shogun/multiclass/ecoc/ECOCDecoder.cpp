/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#include <mathematics/Math.h>
#include <multiclass/ecoc/ECOCDecoder.h>

using namespace shogun;

SGVector<float64_t> CECOCDecoder::binarize(const SGVector<float64_t> query)
{
    SGVector<float64_t> bquery(query.vlen);
    for (int32_t i=0; i < query.vlen; ++i)
    {
        if (query.vector[i] >= 0)
            bquery.vector[i] = +1.0;
        else
            bquery.vector[i] = -1.0;
    }

    return bquery;
}

