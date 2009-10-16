/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "preproc/PreProc.h"
#include "lib/io.h"

using namespace shogun;

CPreProc::CPreProc(const char* name, const char* id)
: CSGObject(), preproc_name(name), preproc_id(id)
{
	SG_INFO("Creating PreProc \"%s\".\n", preproc_name);
}

CPreProc::~CPreProc()
{
	SG_INFO("Deleting PreProc \"%s\".\n", preproc_name);
}
