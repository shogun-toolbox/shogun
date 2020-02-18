/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUNALLOCATOR_H_
#define SHOGUNALLOCATOR_H_

#include <shogun/mathematics/graph/Types.h>

namespace shogun
{
	namespace graph
	{

		inline void* allocator_dispatch(size_t size, element_type type)
		{
#define ALLOCATE_TYPE(TYPE) return new get_type_from_enum<TYPE>::type[size]();
			switch (type)
			{
			case element_type::FLOAT32:
				ALLOCATE_TYPE(element_type::FLOAT32)
			case element_type::FLOAT64:
				ALLOCATE_TYPE(element_type::FLOAT64)
			case element_type::BOOLEAN:
				ALLOCATE_TYPE(element_type::BOOLEAN)
			case element_type::INT8:
				ALLOCATE_TYPE(element_type::INT8)
			case element_type::INT16:
				ALLOCATE_TYPE(element_type::INT16)
			case element_type::INT32:
				ALLOCATE_TYPE(element_type::INT32)
			case element_type::INT64:
				ALLOCATE_TYPE(element_type::INT64)
			case element_type::UINT8:
				ALLOCATE_TYPE(element_type::UINT8)
			case element_type::UINT16:
				ALLOCATE_TYPE(element_type::UINT16)
			case element_type::UINT32:
				ALLOCATE_TYPE(element_type::UINT32)
			case element_type::UINT64:
				ALLOCATE_TYPE(element_type::UINT64)
			}
#undef ALLOCATE_TYPE
		}

		inline void deallocator_dispatch(void* data, element_type type)
		{
#define DEALLOCATE_TYPE(TYPE)                                                  \
	return delete (get_type_from_enum<TYPE>::type*)data;                       \
	break;

			switch (type)
			{
			case element_type::FLOAT32:
				DEALLOCATE_TYPE(element_type::FLOAT32)
			case element_type::FLOAT64:
				DEALLOCATE_TYPE(element_type::FLOAT64)
			case element_type::BOOLEAN:
				DEALLOCATE_TYPE(element_type::BOOLEAN)
			case element_type::INT8:
				DEALLOCATE_TYPE(element_type::INT8)
			case element_type::INT16:
				DEALLOCATE_TYPE(element_type::INT16)
			case element_type::INT32:
				DEALLOCATE_TYPE(element_type::INT32)
			case element_type::INT64:
				DEALLOCATE_TYPE(element_type::INT64)
			case element_type::UINT8:
				DEALLOCATE_TYPE(element_type::UINT8)
			case element_type::UINT16:
				DEALLOCATE_TYPE(element_type::UINT16)
			case element_type::UINT32:
				DEALLOCATE_TYPE(element_type::UINT32)
			case element_type::UINT64:
				DEALLOCATE_TYPE(element_type::UINT64)
			}
#undef DEALLOCATE_TYPE
		}
	} // namespace graph
} // namespace shogun

#endif