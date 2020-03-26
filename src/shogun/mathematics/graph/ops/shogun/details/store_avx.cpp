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
	void store_avx(const aligned_vector& input, void* output);

	template <typename T>
	void store_avx(const aligned_vector& data, void* output)
	{
		using vector_type = typename alignedvector_from_builtintype<T, AVX_BYTESIZE>::type;
		if constexpr(std::is_integral_v<T>)
			_mm256_storeu_si256(static_cast<vector_type*>(output), std::get<vector_type>(data));
		else
			pstoreu(static_cast<T*>(output), std::get<vector_type>(data));
	}

	template void store_avx<bool>(const aligned_vector&, void*);
	template void store_avx<int8_t>(const aligned_vector&, void*);
	template void store_avx<int16_t>(const aligned_vector&, void*);
	template void store_avx<int32_t>(const aligned_vector&, void*);
	template void store_avx<int64_t>(const aligned_vector&, void*);
	template void store_avx<uint8_t>(const aligned_vector&, void*);
	template void store_avx<uint16_t>(const aligned_vector&, void*);
	template void store_avx<uint32_t>(const aligned_vector&, void*);
	template void store_avx<uint64_t>(const aligned_vector&, void*);
	template void store_avx<float>(const aligned_vector&, void*);
	template void store_avx<double>(const aligned_vector&, void*);
}
