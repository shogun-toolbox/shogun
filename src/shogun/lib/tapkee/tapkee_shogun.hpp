#ifndef TAPKEE_SHOGUN_ADAPTER
#define TAPKEE_SHOGUN_ADAPTER

#include <shogun/io/SGIO.h>

using namespace shogun;

#define TAPKEE_EIGEN_INCLUDE_FILE <shogun/mathematics/eigen3.h>
#include <shogun/lib/tapkee/tapkee_defines.hpp>

namespace shogun
{

class ShogunLoggerImplementation : public LoggerImplementation
{
	virtual void message_info(const std::string& msg)
	{
		SG_SINFO(msg.c_str());
	}
	virtual void message_warning(const std::string& msg)
	{
		SG_SWARNING(msg.c_str());
	}
	virtual void message_error(const std::string& msg)
	{
		SG_SERROR(msg.c_str());
	}
	virtual void message_benchmark(const std::string& msg)
	{
		SG_SINFO(msg.c_str());
	}
};

}

#endif 
