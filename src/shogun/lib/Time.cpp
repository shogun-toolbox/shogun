/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <lib/Time.h>
#include <io/SGIO.h>
#include <lib/common.h>

using namespace shogun;

CTime::CTime(bool st)
: CSGObject()
{
	start_time=0;
	stop_time=0;
	start_runtime=0;

	if (st)
		start();
}

CTime::~CTime()
{
}

clock_t CTime::cur_runtime(bool verbose)
{
	clock_t cur_time=clock();
	if (verbose)
		SG_PRINT("current %ld\n", (int64_t) cur_time)
	return cur_time;
}

clock_t CTime::cur_runtime_diff(bool verbose)
{
	clock_t diff=clock()-start_runtime;
	if (verbose)
		SG_PRINT("current diff %ld\n", (int64_t) diff)
	return diff;
}

float64_t CTime::cur_runtime_diff_sec(bool verbose)
{
	float64_t diff_s = ((float64_t)(clock() - start_runtime)) / CLOCKS_PER_SEC;
	if (verbose)
		SG_PRINT("%2.1f seconds\n", diff_s)

	return diff_s;
}


float64_t CTime::start(bool verbose)
{
	start_time=get_curtime();

	if (verbose)
		SG_PRINT("start %ld\n", (int64_t) start_time)
	return start_time;
}

float64_t CTime::cur_time_diff(bool verbose)
{
	float64_t diff_s = get_curtime()-start_time;
	if (verbose)
		SG_PRINT("%2.1f seconds\n", diff_s)

	return diff_s;
}

float64_t CTime::time_diff_sec(bool verbose)
{
	float64_t diff_s = stop_time - start_time;
	if (verbose)
		SG_PRINT("%2.1f seconds\n", diff_s)

	return diff_s;
}

float64_t CTime::stop(bool verbose)
{
	stop_time=get_curtime();

	if (verbose)
		SG_PRINT("stop %ld\n", (int64_t) stop_time)
	return stop_time;
}
