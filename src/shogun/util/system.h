/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal
 */
#ifndef __UTIL_SYSTEM_H__
#define __UTIL_SYSTEM_H__

#include <shogun/lib/common.h>

namespace shogun
{
	namespace utils
	{
		static bool is_big_endian()
		{
			union {
				uint32_t i;
				char c[4];
			} bint = {0x01020304};

			return bint.c[0] == 1;
		}
	} // namespace utils
} // namespace shogun

#endif
