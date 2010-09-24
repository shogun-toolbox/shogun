/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2010 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#include "lib/SerialFile.h"

using namespace shogun;

CSerialFile::CSerialFile(void) :CSGObject()
{
	file = NULL;
	task = 0;
	filename = strdup("(file)");
}

CSerialFile::CSerialFile(FILE* f, char rw) :CSGObject()
{
	file = f;
	task = rw;
	filename = strdup("(file)");
}

CSerialFile::CSerialFile(char* fname, char rw) :CSGObject()
{
	task = rw;
	filename = strdup(fname);
	char mode[2];
	mode[0] = rw;
	mode[1] = '\0';

	if (rw == 'r' || rw == 'w') {
		if (filename) {
			if (!(file=fopen((const char*) filename,
							 (const char*) mode)))
				SG_ERROR("Error opening file '%s'\n", filename);
		}
	} else
		SG_ERROR("unknown mode '%c'\n", mode[0]);
}

CSerialFile::~CSerialFile(void)
{
	close();
}

void
CSerialFile::close()
{
	free(filename);
	if (file)
		fclose(file);
	filename=NULL;
	file=NULL;
}

bool
CSerialFile::is_task_warn(char rw)
{
	if (rw == 'w' && task != 'w') {
		SG_WARNING("`%s' not opened for writing!\n", filename);
		return false;
	}
	if (rw == 'r' && task != 'r') {
		SG_WARNING("`%s' not opened for reading!\n", filename);
		return false;
	}

	return true;
}

bool
CSerialFile::false_warn(const char* prefix, const char* name)
{
	if (task == 'w')
		SG_WARNING("Could not write `%s%s' from `%s'!", prefix,
				   name, filename);
	if (task == 'r')
		SG_WARNING("Could not read `%s%s' from `%s'!", prefix,
				   name, filename);
	if (task != 'w' && task != 'r')
		SG_WARNING("Could not read/write `%s%s' from `%s'!",
				   prefix, name, filename);

	return false;
}
