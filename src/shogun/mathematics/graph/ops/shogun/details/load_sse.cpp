/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include "PacketType.h"
#include <Eigen/Core>

using namespace Eigen::internal;

namespace shogun::graph::op {
			
	template <typename T>
	aligned_vector load_sse(void* input1);

	template <typename T>
	aligned_vector load_sse(void* data)
	{
		if constexpr(std::is_same_v<T, bool>)
			return static_cast<bool*>(data);
		else
		{
			using vector_type = typename alignedvector_from_builtintype<T, SSE_BYTESIZE>::type;
			if constexpr(std::is_integral_v<T>)
				return _mm_loadu_si128(static_cast<const vector_type*>(data));
			else
				return ploadu<vector_type>(static_cast<const T*>(data));
		}
	}

	template aligned_vector load_sse<bool>(void*);
	template aligned_vector load_sse<int8_t>(void*);
	template aligned_vector load_sse<int16_t>(void*);
	template aligned_vector load_sse<int32_t>(void*);
	template aligned_vector load_sse<int64_t>(void*);
	template aligned_vector load_sse<uint8_t>(void*);
	template aligned_vector load_sse<uint16_t>(void*);
	template aligned_vector load_sse<uint32_t>(void*);
	template aligned_vector load_sse<uint64_t>(void*);
	template aligned_vector load_sse<float>(void*);
	template aligned_vector load_sse<double>(void*);
}
