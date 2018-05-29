/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Thoralf Klein, Viktor Gal, Soumyajit De, 
 *          Evangelos Anagnostopoulos
 */

#include <shogun/base/Parallel.h>
#include <shogun/lib/RefCount.h>
#include <shogun/lib/config.h>
#include <shogun/lib/memory.h>

#include <thread>

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
	num_cpus = std::thread::hardware_concurrency();
	// it can return 0, hence only the value
	// if it's correct, else continue with detection
	// of cpus
	if (num_cpus > 0)
		return num_cpus;

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
