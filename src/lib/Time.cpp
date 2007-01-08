/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/Time.h"
#include "lib/io.h"
#include "lib/common.h"

CTime::CTime(bool st)
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
		CIO::message(M_INFO, "current %ld\n", (LONG) cur_time);
	return cur_time;
}

clock_t CTime::cur_runtime_diff(bool verbose)
{
	clock_t diff=clock()-start_runtime;
	if (verbose)
		CIO::message(M_INFO, "current diff %ld\n", (LONG) diff);
	return diff;
}

double CTime::cur_runtime_diff_sec(bool verbose)
{
	double diff_s = ((double)(clock() - start_runtime)) / CLOCKS_PER_SEC;
	if (verbose)
		CIO::message(M_INFO, "%2.1f seconds\n", diff_s);

	return diff_s;
}


double CTime::start(bool verbose)
{
	start_time=get_curtime();

	if (verbose)
		CIO::message(M_INFO, "start %ld\n", (LONG) start_time);
	return start_time;
}

double CTime::cur_time_diff(bool verbose)
{
	double diff_s = get_curtime()-start_time;
	if (verbose)
		CIO::message(M_INFO, "%2.1f seconds\n", diff_s);

	return diff_s;
}

double CTime::time_diff_sec(bool verbose)
{
	double diff_s = stop_time - start_time;
	if (verbose)
		CIO::message(M_INFO, "%2.1f seconds\n", diff_s);

	return diff_s;
}

double CTime::stop(bool verbose)
{
	stop_time=get_curtime();

	if (verbose)
		CIO::message(M_INFO, "stop %ld\n", (LONG) stop_time);
	return stop_time;
}
