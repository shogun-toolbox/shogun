/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Chiyuan Zhang
 */

#include <shogun/multiclass/ecoc/ECOCOVOEncoder.h>

using namespace shogun;

SGMatrix<int32_t> ECOCOVOEncoder::create_codebook(int32_t num_classes)
{
    SGMatrix<int32_t> code_book(num_classes*(num_classes-1)/2, num_classes, true);
    code_book.zero();
    int32_t k=0;
    for (int32_t i=0; i < num_classes; ++i)
    {
        for (int32_t j=i+1; j < num_classes; ++j)
        {
            code_book(k, i) = 1;
            code_book(k, j) = -1;
            k++;
        }
    }

    return code_book;
}

