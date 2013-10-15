/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#ifndef TAPKEE_TIME_H_
#define TAPKEE_TIME_H_

/* Tapkee includes */
#include <shogun/lib/tapkee/utils/logging.hpp>
/* End of Tapkee includes */

#include <ctime>
#include <string>
#include <sstream>

namespace tapkee
{
namespace tapkee_internal
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
	std::string operation_name;
	timed_context(const std::string& name) : start_clock(CLOCK_GET), operation_name(name)
	{
	}
	~timed_context()
	{
		std::stringstream msg_stream;
		msg_stream << operation_name << " took " << double(CLOCK_GET-start_clock)/CLOCK_DIVISOR << " seconds.";
		LoggingSingleton::instance().message_benchmark(msg_stream.str());
	}
};
}
}

#undef CLOCK_TYPE
#undef CLOCK_GET
#undef CLOCK_DIVISOR

#endif
