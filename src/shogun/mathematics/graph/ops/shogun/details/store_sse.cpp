/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include <Eigen/Core>
#include "PacketType.h"

using namespace Eigen::internal;

namespace shogun::graph::op {

	template <typename T>
	void store_sse(const aligned_vector& input, void* output);

	template <typename T>
	void store_sse(const aligned_vector& data, void* output)
	{
		if constexpr(std::is_same_v<T, bool>)
			output = (void*)std::get<bool*>(data);
		else
		{
			using vector_type = typename alignedvector_from_builtintype<T, SSE_BYTESIZE>::type;
			if constexpr(std::is_integral_v<T>)
				_mm_storeu_si128(static_cast<vector_type*>(output), std::get<vector_type>(data));
			else
				pstoreu(static_cast<T*>(output), std::get<vector_type>(data));
		}
	}

	template void store_sse<bool>(const aligned_vector&, void*);
	template void store_sse<int8_t>(const aligned_vector&, void*);
	template void store_sse<int16_t>(const aligned_vector&, void*);
	template void store_sse<int32_t>(const aligned_vector&, void*);
	template void store_sse<int64_t>(const aligned_vector&, void*);
	template void store_sse<uint8_t>(const aligned_vector&, void*);
	template void store_sse<uint16_t>(const aligned_vector&, void*);
	template void store_sse<uint32_t>(const aligned_vector&, void*);
	template void store_sse<uint64_t>(const aligned_vector&, void*);
	template void store_sse<float>(const aligned_vector&, void*);
	template void store_sse<double>(const aligned_vector&, void*);
}
