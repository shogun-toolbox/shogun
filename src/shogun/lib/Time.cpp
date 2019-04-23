/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Thoralf Klein, Viktor Gal
 */

#include <shogun/lib/Time.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/common.h>
#include <shogun/base/SGObject.h>

using namespace shogun;

Time::Time(bool st)
: SGObject()
{
	start_time=0;
	stop_time=0;
	start_runtime=0;

	if (st)
		start();
}

Time::~Time()
{
}

clock_t Time::cur_runtime(bool verbose)
{
	clock_t cur_time=clock();
	if (verbose)
		SG_PRINT("current %ld\n", (int64_t) cur_time)
	return cur_time;
}

clock_t Time::cur_runtime_diff(bool verbose)
{
	clock_t diff=clock()-start_runtime;
	if (verbose)
		SG_PRINT("current diff %ld\n", (int64_t) diff)
	return diff;
}

float64_t Time::cur_runtime_diff_sec(bool verbose)
{
	float64_t diff_s = ((float64_t)(clock() - start_runtime)) / CLOCKS_PER_SEC;
	if (verbose)
		SG_PRINT("%2.1f seconds\n", diff_s)

	return diff_s;
}


float64_t Time::start(bool verbose)
{
	start_time=get_curtime();

	if (verbose)
		SG_PRINT("start %ld\n", (int64_t) start_time)
	return start_time;
}

float64_t Time::cur_time_diff(bool verbose)
{
	float64_t diff_s = get_curtime()-start_time;
	if (verbose)
		SG_PRINT("%2.1f seconds\n", diff_s)

	return diff_s;
}

float64_t Time::time_diff_sec(bool verbose)
{
	float64_t diff_s = stop_time - start_time;
	if (verbose)
		SG_PRINT("%2.1f seconds\n", diff_s)

	return diff_s;
}

float64_t Time::stop(bool verbose)
{
	stop_time=get_curtime();

	if (verbose)
		SG_PRINT("stop %ld\n", (int64_t) stop_time)
	return stop_time;
}

#if defined(_MSC_VER) || defined(__MINGW32__) || defined(__MINGW64__)
#pragma comment(lib, "winmm")
int gettimeofday(struct timeval* tp, void* tzp)
{
    DWORD t;
    t = timeGetTime();
    tp->tv_sec = t / 1000;
    tp->tv_usec = t % 1000;
    /* 0 indicates that the call succeeded. */
    return 0;
}
#endif
