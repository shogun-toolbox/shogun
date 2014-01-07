/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <lib/ShogunException.h>
#include <lib/Signal.h>

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

using namespace shogun;

void
ShogunException::init(const char* str)
{
	size_t n = strlen(str) + 1;

	val = (char*) malloc(n);
	if (val)
		strncpy(val, str, n);
	else {
		fprintf(stderr, "Could not even allocate memory for exception"
				" - dying.\n");
		exit(1);
	}
}

ShogunException::ShogunException(const char* str)
{
#ifndef WIN32
	CSignal::unset_handler();
#endif

	init(str);
}

ShogunException::ShogunException(const ShogunException& orig)
{ init(orig.val); }

ShogunException::~ShogunException() { free(val); }
