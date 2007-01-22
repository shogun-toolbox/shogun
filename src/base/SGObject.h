/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007 Soeren Sonnenburg
 * Copyright (C) 2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __SGOBJECT_H__
#define __SGOBJECT_H__

#include "lib/io.h"
#include "base/Parallel.h"
#include "base/Version.h"

class CSGObject;

class CSGObject
{
public:
#ifdef HAVE_SWIG
	CParallel parallel;
	CIO io;
	CVersion version;
#else
	static CParallel parallel;
	static CIO io;
	static CVersion version;
#endif
};
#endif // __SGOBJECT_H__
