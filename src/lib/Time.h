/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __TIME_H__
#define __TIME_H__

#include <sys/time.h>
#include <time.h>

#include "lib/common.h"
#include "lib/io.h"
#include "base/SGObject.h"

/** Class Time that implements a stopwatch based on either cpu time or wall
 * clock time. */
class CTime : public CSGObject
{
	public:
		/** constructor
		 *
		 * @param start if time measurement shall be started
		 */
		CTime(bool start=true);
		~CTime();

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
		 *
		 * @param verbose if time difference shall be printed
		 * @return time difference between start and NOW in seconds
		 */
		double cur_runtime_diff_sec(bool verbose=false);

		/** start the counter
		 *
		 * @param verbose if start time shall be printed
		 * @return start time in seconds
		 */
		double start(bool verbose=false);

		/** get time difference between start and NOW in seconds
		 *
		 * @param verbose if time difference shall be printed
		 * @return time difference between start and NOW in seconds
		 */
		double cur_time_diff(bool verbose=false);

		/** get time difference between start and stop in seconds
		 *
		 * @param verbose if time difference shall be printed
		 * @return time difference between start and stop in seconds
		 */
		double time_diff_sec(bool verbose=false);

		/** stop the counter
		 * @param verbose if stop time shall be printed
		 * @return stop time in seconds
		 */
		double stop(bool verbose=false);

		/** get runtime
		 *
		 * @return runtime
		 */
		static LONG get_runtime()
		{
			clock_t start_runtime = clock();
			return((LONG)((double)start_runtime*100.0/(double)CLOCKS_PER_SEC));
		}

		/** get current time in seconds
		 *
		 * @return current time in seconds
		 */
		static double get_curtime()
		{
			timeval tv;
			if (gettimeofday(&tv, NULL)==0)
				return tv.tv_sec+((double)(tv.tv_usec))/1e6;
			else
				return 0.0;
		}

	protected:
		/** start runtime */
		clock_t start_runtime;

		/** start time */
		double start_time;
		/** stop time */
		double stop_time;
};
#endif
