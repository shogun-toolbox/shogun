#include "lib/Time.h"
#include "lib/io.h"

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
		CIO::message("start %ld\n", (long) start_time);
	return start_time;
}

clock_t CTime::cur_time_diff(bool verbose)
{
	clock_t cur_time=clock();
	if (verbose)
		CIO::message("current %ld\n", (long) cur_time);
	return cur_time;
}

clock_t CTime::stop(bool verbose)
{
	stop_time=clock();
	if (verbose)
		CIO::message("stop %ld\n", (long) stop_time);
	return stop_time;
}


double CTime::cur_time_diff_sec(bool verbose)
{
	double diff_s = ((double)(clock() - start_time)) / CLOCKS_PER_SEC;
	if (verbose)
		CIO::message( "%2.1f seconds\n", diff_s);

	return diff_s;
}

double CTime::time_diff_sec(bool verbose)
{
	double diff_s = ((double)(stop_time - start_time)) / CLOCKS_PER_SEC;
	if (verbose)
		CIO::message( "%2.1f seconds\n", diff_s);

	return diff_s;
}
