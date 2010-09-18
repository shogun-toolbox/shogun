/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#include "lib/File.h"
#include "features/SparseFeatures.h"
#include "lib/BinaryFile.h"

using namespace shogun;

CBinaryFile::CBinaryFile(FILE* f, char rw) : CFile(f, rw)
{
}

CBinaryFile::CBinaryFile(char* fname, char rw) : CFile(fname, rw)
{
}

CBinaryFile::~CBinaryFile()
{
}
