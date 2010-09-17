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

#include "lib/File.h"

using namespace shogun;

CFile::CFile() :CSGObject()
{
	file = NULL;
	filename = NULL;
}

CFile::CFile(FILE* f, char rw) :CSGObject()
{
	file = f;
	task = rw;
	filename = NULL;
}

CFile::CFile(char* fname, char rw) :CSGObject()
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

CFile::~CFile()
{
	close();
}

void
CFile::close()
{
	free(filename);
	if (file)
		fclose(file);
	filename=NULL;
	file=NULL;
}
