#include "lib/Time.h"
#include "lib/io.h"
#include "lib/common.h"

CTime::CTime(bool st)
{
	if (st)
		start();
	else
		start_time=0;
}

CTime::~CTime()
{
}

clock_t CTime::start(bool verbose)
{
	start_time=clock();
	if (verbose)
		CIO::message(M_INFO, "start %ld\n", (LONG) start_time);
	return start_time;
}

clock_t CTime::cur_time(bool verbose)
{
	clock_t cur_time=clock();
	if (verbose)
		CIO::message(M_INFO, "current %ld\n", (LONG) cur_time);
	return cur_time;
}

clock_t CTime::cur_time_diff(bool verbose)
{
	clock_t cur_time_diff=clock()-start_time;
	if (verbose)
		CIO::message(M_INFO, "current diff %ld\n", (LONG) cur_time_diff);
	return cur_time_diff;
}

clock_t CTime::stop(bool verbose)
{
	stop_time=clock();
	if (verbose)
		CIO::message(M_INFO, "stop %ld\n", (LONG) stop_time);
	return stop_time;
}


double CTime::cur_time_diff_sec(bool verbose)
{
	double diff_s = ((double)(clock() - start_time)) / CLOCKS_PER_SEC;
	if (verbose)
		CIO::message(M_INFO, "%2.1f seconds\n", diff_s);

	return diff_s;
}

double CTime::time_diff_sec(bool verbose)
{
	double diff_s = ((double)(stop_time - start_time)) / CLOCKS_PER_SEC;
	if (verbose)
		CIO::message(M_INFO, "%2.1f seconds\n", diff_s);

	return diff_s;
}
