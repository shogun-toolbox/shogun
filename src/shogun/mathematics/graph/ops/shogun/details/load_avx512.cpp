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
	aligned_vector load_avx512(void* input1);

	template <typename T>
	aligned_vector load_avx512(void* data)
	{
		if constexpr(std::is_same_v<T, bool>)
			return static_cast<bool*>(data);
		else
		{
			using vector_type = typename alignedvector_from_builtintype<T, AVX512_BYTESIZE>::type;
			if constexpr(std::is_integral_v<T>)
				return _mm512_loadu_si512(static_cast<const vector_type*>(data));
			else
				return ploadu<vector_type>(static_cast<const T*>(data));
		}
	}

	template aligned_vector load_avx512<bool>(void*);
	template aligned_vector load_avx512<int8_t>(void*);
	template aligned_vector load_avx512<int16_t>(void*);
	template aligned_vector load_avx512<int32_t>(void*);
	template aligned_vector load_avx512<int64_t>(void*);
	template aligned_vector load_avx512<uint8_t>(void*);
	template aligned_vector load_avx512<uint16_t>(void*);
	template aligned_vector load_avx512<uint32_t>(void*);
	template aligned_vector load_avx512<uint64_t>(void*);
	template aligned_vector load_avx512<float>(void*);
	template aligned_vector load_avx512<double>(void*);
}
