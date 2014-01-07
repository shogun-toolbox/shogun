/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#include <multiclass/ecoc/ECOCOVREncoder.h>

using namespace shogun;

SGMatrix<int32_t> CECOCOVREncoder::create_codebook(int32_t num_classes)
{
    SGMatrix<int32_t> code_book(num_classes, num_classes, true);
    code_book.set_const(-1);
    for (int32_t i=0; i < num_classes; ++i)
        code_book(i, i) = 1;

    return code_book;
}
