/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Chiyuan Zhang, Soeren Sonnenburg
 */

#include <shogun/mathematics/Math.h>
#include <shogun/multiclass/ecoc/ECOCDecoder.h>

using namespace shogun;

SGVector<float64_t> ECOCDecoder::binarize(const SGVector<float64_t> query)
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

