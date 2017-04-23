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
#include <shogun/lib/RefCount.h>
#include <shogun/lib/config.h>
#include <shogun/lib/memory.h>

#ifdef HAVE_CXX11
#include <thread>
#endif

#if defined(LINUX)
#include <unistd.h>
#elif defined(DARWIN)
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

#ifdef HAVE_OPENMP
#include <omp.h>
#endif

using namespace shogun;

Parallel::Parallel()
{
	num_threads=get_num_cpus();
	m_refcount = new RefCount();
#ifdef HAVE_OPENMP
	omp_set_dynamic(0);
	omp_set_num_threads(num_threads);
#endif
}

Parallel::Parallel(const Parallel& orig)
{
	num_threads=orig.get_num_threads();
	m_refcount = new RefCount();
#ifdef HAVE_OPENMP
	omp_set_dynamic(0);
	omp_set_num_threads(num_threads);
#endif
}

Parallel::~Parallel()
{
	delete m_refcount;
}

int32_t Parallel::get_num_cpus() const
{
	int32_t num_cpus = 1;
#if HAVE_CXX11
	num_cpus = std::thread::hardware_concurrency();
	// it can return 0, hence only the value
	// if it's correct, else continue with detection
	// of cpus
	if (num_cpus > 0)
		return num_cpus;
#endif

#if defined(LINUX) && defined(_SC_NPROCESSORS_ONLN)
	return sysconf(_SC_NPROCESSORS_ONLN);
#elif defined(DARWIN)
	int32_t num; /* for calling external lib */
	size_t size=sizeof(num);
	if (!sysctlbyname("hw.ncpu", &num, &size, NULL, 0))
		num_cpus = num;
#elif defined(_MSC_VER)
	SYSTEM_INFO sysinfo;
	GetSystemInfo(&sysinfo);
	num_cpus = sysinfo.dwNumberOfProcessors;
#endif

	return num_cpus;
}

void Parallel::set_num_threads(int32_t n)
{
#if !defined(HAVE_PTHREAD) && !defined(HAVE_OPENMP)
	ASSERT(n==1)
#endif
	num_threads=n;
#ifdef HAVE_OPENMP
	omp_set_num_threads(num_threads);
#endif
}

int32_t Parallel::get_num_threads() const
{
	return num_threads;
}

int32_t Parallel::ref()
{
	return m_refcount->ref();
}

int32_t Parallel::ref_count() const
{
	return m_refcount->ref_count();
}

int32_t Parallel::unref()
{
	int32_t rc = m_refcount->unref();

	if (rc==0)
	{
		delete this;
		return 0;
	}

	return rc;
}
