/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Giovanni De Toni
 *
 */

#ifndef SHOGUN_UTILS_H_H
#define SHOGUN_UTILS_H_H

#include <shogun/lib/common.h>
#include <limits>
#include <typeinfo>

namespace shogun {

	/**
	 * Cast an input size_t value to the int32_t type.
	 * Throws an exception if something goes wrong.
	 * @code
	 * unsigned long o = 1000000000000;
	 * try {
	 *	int a = size_t_to_int32_cast(o);
	 * } catch (std::exception e)
	 * {
	 *	std::cout << e.what() << std::endl;
	 * }
	 * @endcode
	 * @param v original value
	 * @return casted value
	 */
	SG_FORCED_INLINE int32_t size_t_to_int32_cast(size_t v) {
		if (v > std::numeric_limits<int32_t >::max()) {
			throw std::overflow_error(
					"Conversion error from type size_t to type int32_t");
		}
		return static_cast<int32_t>(v);
	}
}

#endif //SHOGUN_UTILS_H_H
