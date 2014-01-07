/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#include <multiclass/ecoc/ECOCSimpleDecoder.h>

using namespace shogun;

int32_t CECOCSimpleDecoder::decide_label(const SGVector<float64_t> outputs, const SGMatrix<int32_t> codebook)
{
    SGVector<float64_t> query=outputs;

    if (binary_decoding())
        query = binarize(outputs);

    SGVector<float64_t> distances(codebook.num_cols);
    for (int32_t i=0; i < distances.vlen; ++i)
        distances[i] = compute_distance(query, codebook.get_column_vector(i));

    int32_t result = SGVector<float64_t>::arg_min(distances.vector, 1, distances.vlen);
    return result;
}
