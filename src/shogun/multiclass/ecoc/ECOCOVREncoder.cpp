/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Chiyuan Zhang
 */

#include <shogun/multiclass/ecoc/ECOCOVREncoder.h>

using namespace shogun;

SGMatrix<int32_t> ECOCOVREncoder::create_codebook(int32_t num_classes)
{
    SGMatrix<int32_t> code_book(num_classes, num_classes, true);
    code_book.set_const(-1);
    for (int32_t i=0; i < num_classes; ++i)
        code_book(i, i) = 1;

    return code_book;
}
