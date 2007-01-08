/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef PARALLEL_H__
#define PARALLEL_H__

#include "lib/common.h"
#include "lib/config.h"
#include "lib/io.h"

#include <unistd.h>

class CParallel
{
public:
	CParallel();
	~CParallel();

#ifdef HAVE_NPROCESSORS_ONLN
	static inline INT get_num_cpus()
	{
		return sysconf( _SC_NPROCESSORS_ONLN );
	}
#endif

	static inline void set_num_threads(INT n)
	{
#ifdef WIN32
		ASSERT(n==1);
#endif
		num_threads=n;
	}

	static inline INT get_num_threads()
	{
		return num_threads;
	}

protected:
	static INT num_threads;
};
#endif
