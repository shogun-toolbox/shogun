/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef PARALLEL_H__
#define PARALLEL_H__

#include <shogun/lib/common.h>
#include <shogun/lib/config.h>
#include <shogun/io/SGIO.h>

#if defined(LINUX) && defined(_SC_NPROCESSORS_ONLN)
#include <unistd.h>
#elif defined(DARWIN)
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

namespace shogun
{
/** @brief Class Parallel provides helper functions for multithreading.
 *
 * For example it can be used to determine the number of CPU cores in your
 * computer and is the place where you define the number of CPUs that shall be
 * used in computations.
 */
class Parallel
{
public:
	/** constructor */
	Parallel();

	/** copy constructor */
	Parallel(const Parallel& orig);

	/** destructor */
	virtual ~Parallel();

	/** get num of cpus
	 * @return number of CPUs
	 */
	inline int32_t get_num_cpus() const
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

	/** set number of threads
	 * @param n number of threads 
	 */
	inline void set_num_threads(int32_t n)
	{
#ifdef WIN32
		ASSERT(n==1);
#endif
		num_threads=n;
	}

	/** get number of threads 
	 * @return number of threads
	 */
	inline int32_t get_num_threads() const
	{
		return num_threads;
	}

	/** ref
	 * @return current ref counter
	 */
	inline int32_t ref()
	{
		++refcount;
		return refcount;
	}

	/** get ref count
	 * @return current ref counter
	 */
	inline int32_t ref_count() const
	{
		return refcount;
	}

	/** unref
	 * @return current ref counter
	 */
	inline int32_t unref()
	{
		if (refcount==0 || --refcount==0)
		{
			delete this;
			return 0;
		}
		else
			return refcount;
	}

private:

	/** ref counter */
	int32_t refcount;

	/** number of threads */
	int32_t num_threads;
};
}
#endif
