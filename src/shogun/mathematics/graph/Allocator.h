/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUNALLOCATOR_H_
#define SHOGUNALLOCATOR_H_

#include <shogun/mathematics/graph/Types.h>


namespace shogun {
	void* allocator_dispatch(size_t size, element_type type)
	{
		switch (type)
		{
		case element_type::FLOAT32:
			return new get_type_from_enum<element_type::FLOAT32>::type(size);
		case element_type::FLOAT64:
			return new get_type_from_enum<element_type::FLOAT64>::type(size);
		}
	}
}

#endif