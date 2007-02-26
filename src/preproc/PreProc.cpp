/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Written (W) 1999-2007 Gunnar Raetsch
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "preproc/PreProc.h"
#include "lib/io.h"

CPreProc::CPreProc(const CHAR* name, const CHAR* id)
  : CSGObject(), preproc_name(name), preproc_id(id)
{
	SG_INFO( "creating PreProc \"%s\"\n", preproc_name);
}

CPreProc::~CPreProc()
{
  SG_INFO( "deleting PreProc \"%s\"\n", preproc_name) ;
}
