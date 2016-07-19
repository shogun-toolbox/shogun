#ifndef __LBFGSCOMMON_H__
#define __LBFGSCOMMON_H__

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
  

namespace shogun
{

/** linesearch method */
enum ELBFGSLineSearch
{
	MORETHUENTE=0,
	BACKTRACKING_ARMIJO=1,
	BACKTRACKING_WOLFE=2,
	BACKTRACKING_STRONG_WOLFE=3
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
struct LBFGSLineSearchHelper
{
	static int8_t get_lbfgs_linear_search_id(ELBFGSLineSearch method)
	{
		return (int8_t) method;
	}

	static ELBFGSLineSearch get_lbfgs_linear_search(int8_t method_id)
	{
		REQUIRE(method_id>=0 && method_id<4, "Unsupported method id (%d)\n", method_id);
		return (ELBFGSLineSearch) method_id;
	}
};
#endif

} // namespace shogun

#endif/*__LBFGSCOMMON_H__*/
