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
	void store_avx512(const aligned_vector& input, void* output);

	template <typename T>
	void store_avx512(const aligned_vector& data, void* output)
	{
		if constexpr(std::is_same_v<T, bool>)
			output = (void*)std::get<bool*>(data);
		else
		{
			using vector_type = typename alignedvector_from_builtintype<T, AVX512_BYTESIZE>::type;
			if constexpr(std::is_integral_v<T>)
				_mm512_storeu_si512(static_cast<vector_type*>(output), std::get<vector_type>(data));
			else
				pstoreu(static_cast<T*>(output), std::get<vector_type>(data));
		}
	}

	template void store_avx512<bool>(const aligned_vector&, void*);
	template void store_avx512<int8_t>(const aligned_vector&, void*);
	template void store_avx512<int16_t>(const aligned_vector&, void*);
	template void store_avx512<int32_t>(const aligned_vector&, void*);
	template void store_avx512<int64_t>(const aligned_vector&, void*);
	template void store_avx512<uint8_t>(const aligned_vector&, void*);
	template void store_avx512<uint16_t>(const aligned_vector&, void*);
	template void store_avx512<uint32_t>(const aligned_vector&, void*);
	template void store_avx512<uint64_t>(const aligned_vector&, void*);
	template void store_avx512<float>(const aligned_vector&, void*);
	template void store_avx512<double>(const aligned_vector&, void*);
}
