/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#include <multiclass/ecoc/ECOCOVOEncoder.h>

using namespace shogun;

SGMatrix<int32_t> CECOCOVOEncoder::create_codebook(int32_t num_classes)
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

