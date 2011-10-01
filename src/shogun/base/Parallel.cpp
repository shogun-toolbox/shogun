/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/base/Parallel.h>

#if defined(LINUX) && defined(_SC_NPROCESSORS_ONLN)
#include <unistd.h>
#elif defined(DARWIN)
#include <sys/types.h>
#include <sys/sysctl.h>
#endif


using namespace shogun;

Parallel::Parallel() : refcount(0)
{
	num_threads=get_num_cpus();
}

Parallel::Parallel(const Parallel& orig) : refcount(0)
{
	num_threads=orig.get_num_threads();
}

Parallel::~Parallel()
{
}

int32_t Parallel::get_num_cpus() const
{
#if defined(LINUX) && defined(_SC_NPROCESSORS_ONLN)
		return sysconf( _SC_NPROCESSORS_ONLN );
#elif defined(DARWIN)
		int num; /* for calling external lib */
		size_t size=sizeof(num);
		if (!sysctlbyname("hw.ncpu", &num, &size, NULL, 0))
			return num;
#endif
		return 1;
}

void Parallel::set_num_threads(int32_t n)
{
#ifndef HAVE_PTHREAD
	ASSERT(n==1);
#endif
	num_threads=n;
}

int32_t Parallel::get_num_threads() const
{
	return num_threads;
}

int32_t Parallel::ref()
{
	++refcount;
	return refcount;
}

int32_t Parallel::ref_count() const
{
	return refcount;
}

int32_t Parallel::unref()
{
	if (refcount==0 || --refcount==0)
	{
		delete this;
		return 0;
	}
	else
		return refcount;
}
