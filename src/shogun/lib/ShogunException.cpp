/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Giovanni De Toni
 */

#include <shogun/lib/ShogunException.h>
#include <shogun/lib/Signal.h>

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
#endif

	init(str);
}

ShogunException::ShogunException(const ShogunException& orig)
{ init(orig.val); }

ShogunException::~ShogunException() { free(val); }
