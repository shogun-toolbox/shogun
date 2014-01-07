/*
 * Copyright (c) 2009 Yahoo! Inc.  All rights reserved.  The copyrights
 * embodied in the content of this file are licensed under the BSD
 * (revised) open source license.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Adaptation of Vowpal Wabbit v5.1.
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society.
 */

#include <classifier/vw/cache/VwCacheReader.h>

using namespace shogun;

CVwCacheReader::CVwCacheReader()
	: CSGObject()
{
	fd = -1;
	env = NULL;
}

CVwCacheReader::CVwCacheReader(char * fname, CVwEnvironment* env_to_use)
	: CSGObject()
{
	fd = open(fname, O_RDONLY);

	if (fd < 0)
		SG_SERROR("Error opening the file %s for reading from cache!\n")

	env = env_to_use;
	SG_REF(env);
}

CVwCacheReader::CVwCacheReader(int32_t f, CVwEnvironment* env_to_use)
	: CSGObject()
{
	fd = f;
	env = env_to_use;
	SG_REF(env);
}

CVwCacheReader::~CVwCacheReader()
{
	// Does not attempt to close file as it could have been passed
	// from oustide
	if (env)
		SG_UNREF(env);
}

void CVwCacheReader::set_file(int32_t f)
{
	fd = f;
}

void CVwCacheReader::set_env(CVwEnvironment* env_to_use)
{
	env = env_to_use;
	SG_REF(env);
}

CVwEnvironment* CVwCacheReader::get_env()
{
	SG_REF(env);
	return env;
}
