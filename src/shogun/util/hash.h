/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal
 */

#ifndef __UTIL_HASH_H__
#define __UTIL_HASH_H__

#include <shogun/base/SGObject.h>

namespace std
{
	template<> struct hash<shogun::SGObject>
	{
		typedef shogun::SGObject argument_type;
		typedef std::size_t result_type;

		result_type operator()(const argument_type& o) const
		{
			auto params = o.get_params();
			result_type seed = 0;
			for (const auto& p: params)
				seed ^= p.second->get_value().hash() + 0x9e3779b9 + (seed<<6) + (seed>>2);

			return seed;
		}
	};
}

#endif
