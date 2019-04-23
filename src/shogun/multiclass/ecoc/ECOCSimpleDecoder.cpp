/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sanuj Sharma, Chiyuan Zhang
 */

#include <shogun/multiclass/ecoc/ECOCSimpleDecoder.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

int32_t ECOCSimpleDecoder::decide_label(const SGVector<float64_t> outputs, const SGMatrix<int32_t> codebook)
{
    SGVector<float64_t> query=outputs;

    if (binary_decoding())
        query = binarize(outputs);

    SGVector<float64_t> distances(codebook.num_cols);
    for (int32_t i=0; i < distances.vlen; ++i)
        distances[i] = compute_distance(query, codebook.get_column_vector(i));

    int32_t result = Math::arg_min(distances.vector, 1, distances.vlen);
    return result;
}
