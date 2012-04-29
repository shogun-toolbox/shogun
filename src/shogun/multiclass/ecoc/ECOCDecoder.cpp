/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#include <shogun/mathematics/Math.h>
#include <shogun/multiclass/ecoc/ECOCDecoder.h>

using namespace shogun;

int32_t CECOCDecoder::decide_label(const SGVector<float64_t> &outputs, const SGMatrix<int32_t> &codebook)
{
    SGVector<float64_t> query;
    if (binary_decoding())
    {
        query.vector = SG_MALLOC(float64_t, outputs.vlen);
        query.vlen = outputs.vlen;
        query.do_free = true;
        for (int32_t i=0; i < outputs.vlen; ++i)
        {
            if (outputs.vector[i] >= 0)
                query.vector[i] = +1.0;
            else
                query.vector[i] = -1.0;
        }
    }
    else
    {
        query.vector = outputs.vector;
        query.vlen = outputs.vlen;
        query.do_free = false;
    }

    SGVector<float64_t> distances(codebook.num_cols);
    for (int32_t i=0; i < distances.vlen; ++i)
        distances[i] = compute_distance(query, codebook.get_column_vector(i));

    int32_t result = CMath::arg_min(distances.vector, 1, distances.vlen);
    distances.destroy_vector();
    query.free_vector();
    return result;
}
