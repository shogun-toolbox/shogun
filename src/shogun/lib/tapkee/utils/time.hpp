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
#include "logging.hpp"

using std::string;
using std::stringstream;

struct timed_context
{
	clock_t start_clock;
	string operation_name;
	timed_context(const std::string& name) : start_clock(clock()), operation_name(name)
	{
	}
	~timed_context()
	{
		stringstream msg_stream;
		msg_stream << operation_name << " took " << double(clock()-start_clock)/CLOCKS_PER_SEC << " seconds."; 
		LoggingSingleton::instance().message_benchmark(msg_stream.str());
	}
};
#endif
