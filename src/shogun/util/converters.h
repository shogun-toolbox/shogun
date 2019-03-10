/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal
 */
#ifndef __UTIL_CONVERTERS_H__
#define __UTIL_CONVERTERS_H__

#include <shogun/lib/common.h>

namespace shogun
{
	template<typename O, typename I>
	O safe_convert(I i)
	{
		if (i > static_cast<I>((std::numeric_limits<O>::max)()))
		{
			throw std::overflow_error(
				"value cannot be stored in a variable of type requested.");
		}
		return static_cast<O>(i);
	}
}

#endif
