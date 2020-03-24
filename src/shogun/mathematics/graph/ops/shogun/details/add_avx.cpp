/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

// #include <unsupported/Eigen/CXX11/Tensor>

#include <algorithm>
#include <functional>
#include <immintrin.h>

namespace shogun::graph::op {
	template <typename T1, typename T2=T1>
	void add_kernel_implementation_avx(
	    const T2* input1, const T2* input2, T2* output);

	template <typename T1, typename T2>
	void add_kernel_implementation_avx(
	    const T2* input1, const T2* input2, T2* output)
	{
		if constexpr (std::is_same_v<T1, float>)
			*static_cast<__m256*>(output) = _mm256_add_ps(*static_cast<const __m256*>(input1), *static_cast<const __m256*>(input2));
		else
		{
			std::transform(
			    static_cast<const T2*>(input1),
			    static_cast<const T2*>(input1) + 32/sizeof(T2),
			    static_cast<const T2*>(input2), static_cast<T2*>(output),
			    std::plus<T2>());
		}
	}

	template void add_kernel_implementation_avx<bool>(const bool*, const bool*, bool*);
	template void add_kernel_implementation_avx<int8_t>(const int8_t*, const int8_t*, int8_t*);
	template void add_kernel_implementation_avx<int16_t>(const int16_t*, const int16_t*, int16_t*);
	template void add_kernel_implementation_avx<int32_t>(const int32_t*, const int32_t*, int32_t*);
	template void add_kernel_implementation_avx<int64_t>(const int64_t*, const int64_t*, int64_t*);
	template void add_kernel_implementation_avx<uint8_t>(const uint8_t*, const uint8_t*, uint8_t*);
	template void add_kernel_implementation_avx<uint16_t>(const uint16_t*, const uint16_t*, uint16_t*);
	template void add_kernel_implementation_avx<uint32_t>(const uint32_t*, const uint32_t*, uint32_t*);
	template void add_kernel_implementation_avx<uint64_t>(const uint64_t*, const uint64_t*, uint64_t*);
	template void add_kernel_implementation_avx<float>(const void*, const void*, void*);
	template void add_kernel_implementation_avx<double>(const double*, const double*, double*);
}