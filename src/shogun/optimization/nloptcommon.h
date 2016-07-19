#ifndef NLOPTCOMMON_H
#define NLOPTCOMMON_H

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>

#ifdef USE_GPL_SHOGUN
#ifdef HAVE_NLOPT
#include <nlopt.h>
#endif 

namespace shogun
{

#ifdef HAVE_NLOPT
#ifndef DOXYGEN_SHOULD_SKIP_THIS
struct NLOPTAlgorithmHelper
{
	static int16_t get_nlopt_algorithm_id(nlopt_algorithm method)
	{
		return (int16_t) method;
	}

	static nlopt_algorithm get_nlopt_algorithm(int16_t method_id)
	{
		REQUIRE(method_id>=0 && method_id<(int16_t)NLOPT_NUM_ALGORITHMS,
			"Unsupported method id (%d)\n", method_id);
		return (nlopt_algorithm) method_id;
	}
};
#endif
#endif //HAVE_NLOPT

} // namespace shogun

#endif //USE_GPL_SHOGUN
#endif //NLOPTCOMMON_H
