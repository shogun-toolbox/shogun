/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012, Sergey Lisitsyn
 */

#ifndef TAPKEE_TIME_H_
#define TAPKEE_TIME_H_
#include <ctime>
#include <string>
#include <sstream>
#include <utils/logging.hpp>

using std::string;
using std::stringstream;

namespace tapkee
{

#ifdef _OPENMP
	#define CLOCK_TYPE double
	#define CLOCK_GET omp_get_wtime()
	#define CLOCK_DIVISOR 1
#else
	#define CLOCK_TYPE clock_t
	#define CLOCK_GET clock()
	#define CLOCK_DIVISOR CLOCKS_PER_SEC
#endif

struct timed_context
{
	CLOCK_TYPE start_clock;
	string operation_name;
	timed_context(const std::string& name) : start_clock(CLOCK_GET), operation_name(name)
	{
	}
	~timed_context()
	{
		stringstream msg_stream;
		msg_stream << operation_name << " took " << double(CLOCK_GET-start_clock)/CLOCK_DIVISOR << " seconds."; 
		LoggingSingleton::instance().message_benchmark(msg_stream.str());
	}
};

}
#endif
