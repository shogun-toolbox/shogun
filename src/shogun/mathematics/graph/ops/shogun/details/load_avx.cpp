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
	aligned_vector load_avx(void* input1);

	template <typename T>
	aligned_vector load_avx(void* data)
	{
		if constexpr(std::is_same_v<T, bool>)
			return static_cast<bool*>(data);
		else
		{
			using vector_type = typename alignedvector_from_builtintype<T, AVX_BYTESIZE>::type;
			if constexpr(std::is_integral_v<T>)
				return _mm256_loadu_si256(static_cast<const vector_type*>(data));
			else
				return ploadu<vector_type>(static_cast<const T*>(data));
		}
	}

	template aligned_vector load_avx<bool>(void*);
	template aligned_vector load_avx<int8_t>(void*);
	template aligned_vector load_avx<int16_t>(void*);
	template aligned_vector load_avx<int32_t>(void*);
	template aligned_vector load_avx<int64_t>(void*);
	template aligned_vector load_avx<uint8_t>(void*);
	template aligned_vector load_avx<uint16_t>(void*);
	template aligned_vector load_avx<uint32_t>(void*);
	template aligned_vector load_avx<uint64_t>(void*);
	template aligned_vector load_avx<float>(void*);
	template aligned_vector load_avx<double>(void*);
}
