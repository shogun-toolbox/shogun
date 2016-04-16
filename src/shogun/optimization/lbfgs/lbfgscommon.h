#ifndef __LBFGSCOMMON_H__
#define __LBFGSCOMMON_H__

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>

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


} // namespace shogun

#endif/*__LBFGSCOMMON_H__*/
