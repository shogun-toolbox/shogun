/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#include "lib/HDF5File.h"

#ifdef HAVE_HDF5
#include <hdf5.h>

using namespace shogun;

CHDF5File::CHDF5File(void) :CFile()
{
}

CHDF5File::CHDF5File(FILE* f, char rw) : CFile(f, rw)
{
}

CHDF5File::CHDF5File(char* fname, char rw) : CFile(fname, rw)
{
}

CHDF5File::~CHDF5File(void)
{
}

bool
CHDF5File::write_type(const TSGDataType* type, const void* param,
					  const char* name, const char* prefix)
{
	if (!is_task_warn('w')) return false_warn(prefix, name);

	SG_PRINT("writing: %s %s\n", prefix, name);

	return true;
}

bool
CHDF5File::read_type(const TSGDataType* type, void* param,
					 const char* name, const char* prefix)
{
	if (!is_task_warn('r')) return false;

	SG_PRINT("reading: %s %s\n", prefix, name);

	return true;
}

#endif //  HDF5
