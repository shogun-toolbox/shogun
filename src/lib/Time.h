/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __TIME_H__
#define __TIME_H__

#include <sys/time.h>
#include <time.h>

#include "lib/common.h"
#include "lib/io.h"
#include "base/SGObject.h"

class CTime : public CSGObject
{
	public:
		CTime(bool start=true);
		~CTime();

		/// returns the cpu time
		clock_t cur_runtime(bool verbose=false);

		/// returns time difference between start and NOW
		/// when verbose is set the time difference is printed
		clock_t cur_runtime_diff(bool verbose=false);

		/// returns time difference between start and NOW
		/// when verbose is set the time difference is printed
		double cur_runtime_diff_sec(bool verbose=false);

		/// starts counter and returns start time in seconds
		/// when verbose is set the start time is printed
		double start(bool verbose=false);

		/// returns time difference between start and NOW in seconds
		/// when verbose is set the time difference is printed
		double cur_time_diff(bool verbose=false);

		/// returns time difference between start and stop in seconds
		/// when verbose is set the time difference is printed
		double time_diff_sec(bool verbose=false);

		/// stops counter and returns stop time in seconds
		/// when verbose is set the stop time is printed
		double stop(bool verbose=false);

		static LONG get_runtime() 
		{
			clock_t start_runtime = clock();
			return((LONG)((double)start_runtime*100.0/(double)CLOCKS_PER_SEC));
		}

		/// return current time in seconds
		static double get_curtime() 
		{
			timeval tv;
			if (gettimeofday(&tv, NULL)==0)
				return tv.tv_sec+((double)(tv.tv_usec))/1e6;
			else
				return 0.0;
		}

	protected:
		clock_t start_runtime;

		double start_time;
		double stop_time;
};
#endif
