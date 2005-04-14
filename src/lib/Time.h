#ifndef __TIME_H__
#define __TIME_H__

#include <time.h>

#include "lib/common.h"

class CTime
{
	public:
		CTime(bool start=true);
		~CTime();


		/// starts counter and returns start time
		/// when verbose is set the start time is printed
		clock_t start(bool verbose=false);

		/// returns time difference between start and NOW
		/// when verbose is set the time difference is printed
		clock_t cur_time_diff(bool verbose=false);

		/// stops counter and returns stop time
		/// when verbose is set the stop time is printed
		clock_t stop(bool verbose=false);

		/// returns time difference between start and NOW
		/// when verbose is set the time difference is printed
		double cur_time_diff_sec(bool verbose=false);

		/// returns time difference between start and stop
		/// when verbose is set the time difference is printed
		double time_diff_sec(bool verbose=false);

		static LONG get_runtime() 
		{
			clock_t start;
			start = clock();
			return((LONG)((double)start*100.0/(double)CLOCKS_PER_SEC));
		}

	protected:
		clock_t start_time;
		clock_t stop_time;
};
#endif
