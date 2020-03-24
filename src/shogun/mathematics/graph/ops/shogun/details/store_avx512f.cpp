/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include <Eigen/Core>

using namespace Eigen::internal;

#ifndef EIGEN_VECTORIZE_AVX512
static_assert(false, "error");
#endif	

namespace shogun::graph::op {
	template <typename T>
	void store_avx512f(void* input, void* output);

	template <typename T>
	void store_avx512f(void* data, void* output)
	{
		output = data;
	}

	template <>
	void store_avx512f<float>(void* data, void* output)
	{
		pstoreu(static_cast<float*>(output), *static_cast<const Packet16f*>(data));
	}

	template <>
	void store_avx512f<double>(void* data, void* output)
	{
		pstoreu(static_cast<double*>(output), *static_cast<const Packet8d*>(data));
	}

	template <>
	void store_avx512f<int>(void* data, void* output)
	{
		pstoreu(static_cast<int*>(output), *static_cast<const Packet16i*>(data));
	}

	template void store_avx512f<bool>(void*, void*);
	template void store_avx512f<int8_t>(void*, void*);
	template void store_avx512f<int16_t>(void*, void*);
	template void store_avx512f<int32_t>(void*, void*);
	template void store_avx512f<int64_t>(void*, void*);
	template void store_avx512f<uint8_t>(void*, void*);
	template void store_avx512f<uint16_t>(void*, void*);
	template void store_avx512f<uint32_t>(void*, void*);
	template void store_avx512f<uint64_t>(void*, void*);
	template void store_avx512f<float>(void*, void*);
	template void store_avx512f<double>(void*, void*);
}
