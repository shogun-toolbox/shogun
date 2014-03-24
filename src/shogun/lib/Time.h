/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __TIME_H__
#define __TIME_H__

#ifndef _WIN32
#include <sys/time.h>
#endif
#include <time.h>

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/base/SGObject.h>

#if defined(_MSC_VER) || defined(__MINGW32__)

#ifndef _TIMEVAL_DEFINED /* also in winsock[2].h */
#define _TIMEVAL_DEFINED
struct timeval {
    long tv_sec;
    long tv_usec;
};
#endif /* _TIMEVAL_DEFINED */

int gettimeofday(struct timeval* tp, void* tzp) {
    DWORD t;
    t = timeGetTime();
    tp->tv_sec = t / 1000;
    tp->tv_usec = t % 1000;
    /* 0 indicates that the call succeeded. */
    return 0;
}
#endif
namespace shogun
{
/** @brief Class Time that implements a stopwatch based on either cpu time or wall
 * clock time. */
class CTime : public CSGObject
{
	public:
		/** constructor
		 *
		 * @param start if time measurement shall be started
		 */
		CTime(bool start=true);
		virtual ~CTime();

		/** get current cpu runtime
		 *
		 * @param verbose if time shall be printed
		 * @return current cpu runtime
		 */
		clock_t cur_runtime(bool verbose=false);

		/** get time difference between start and NOW
		 *
		 * @param verbose if time difference shall be printed
		 * @return time difference between start and NOW
		 */
		clock_t cur_runtime_diff(bool verbose=false);

		/** get time difference between start and NOW in seconds
		 * (wall clock time)
		 *
		 * @param verbose if time difference shall be printed
		 * @return time difference between start and NOW in seconds
		 */
		float64_t cur_runtime_diff_sec(bool verbose=false);

		/** start the counter
		 *
		 * @param verbose if start time shall be printed
		 * @return start time in seconds
		 */
		float64_t start(bool verbose=false);

		/** get time difference between start and NOW in seconds
		 * (wall clock time)
		 *
		 * @param verbose if time difference shall be printed
		 * @return time difference between start and NOW in seconds
		 */
		float64_t cur_time_diff(bool verbose=false);

		/** get time difference between start and stop in seconds
		 *
		 * @param verbose if time difference shall be printed
		 * @return time difference between start and stop in seconds
		 */
		float64_t time_diff_sec(bool verbose=false);

		/** stop the counter
		 * @param verbose if stop time shall be printed
		 * @return stop time in seconds
		 */
		float64_t stop(bool verbose=false);

		/** get runtime
		 *
		 * @return runtime
		 */
		static float64_t get_runtime()
		{
			clock_t start_runtime = clock();
			return ((float64_t) start_runtime)/CLOCKS_PER_SEC;
		}

		/** get current time in seconds
		 *
		 * @return current time in seconds
		 */
		static float64_t get_curtime()
		{
			timeval tv;
			if (gettimeofday(&tv, NULL)==0)
				return tv.tv_sec+(tv.tv_usec*1e-6);
			else
				return 0.0;
		}

		/** @return object name */
		virtual const char* get_name() const { return "Time"; }

	protected:
		/** start runtime */
		clock_t start_runtime;

		/** start time */
		float64_t start_time;
		/** stop time */
		float64_t stop_time;
};
}
#endif
