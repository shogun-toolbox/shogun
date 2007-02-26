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

#if defined(LINUX) && defined(_SC_NPROCESSORS_ONLN)
#include <unistd.h>
#elif defined(DARWIN)
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

class CParallel
{
public:
	CParallel();
	CParallel(const CParallel& orig);
	~CParallel();

	inline INT get_num_cpus() const
	{
#if defined(LINUX) && defined(_SC_NPROCESSORS_ONLN)
		return sysconf( _SC_NPROCESSORS_ONLN );
#elif defined(DARWIN)
		int num;
		size_t size=sizeof(num);
		if (!sysctlbyname("hw.ncpu", &num, &size, NULL, 0))
			return num;
#endif
		return 1;
	}

	inline void set_num_threads(INT n)
	{
#ifdef WIN32
		ASSERT(n==1);
#endif
		num_threads=n;
	}

	inline INT get_num_threads() const
	{
		return num_threads;
	}

protected:
	INT num_threads;
};
#endif
