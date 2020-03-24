/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include <Eigen/Core>

using namespace Eigen::internal;

namespace shogun::graph::op {
	template <typename T>
	void store_sse(void* input, void* output);

	template <typename T>
	void store_sse(void* data, void* output)
	{
		output = data;
	}

	template <>
	void store_sse<float>(void* data, void* output)
	{
		pstoreu(static_cast<float*>(output), *static_cast<const Packet4f*>(data));
	}

	template <>
	void store_sse<double>(void* data, void* output)
	{
		pstoreu(static_cast<double*>(output), *static_cast<const Packet2d*>(data));
	}

	template <>
	void store_sse<int>(void* data, void* output)
	{
		pstoreu(static_cast<int*>(output), *static_cast<const Packet4i*>(data));
	}

	template void store_sse<bool>(void*, void*);
	template void store_sse<int8_t>(void*, void*);
	template void store_sse<int16_t>(void*, void*);
	template void store_sse<int32_t>(void*, void*);
	template void store_sse<int64_t>(void*, void*);
	template void store_sse<uint8_t>(void*, void*);
	template void store_sse<uint16_t>(void*, void*);
	template void store_sse<uint32_t>(void*, void*);
	template void store_sse<uint64_t>(void*, void*);
	template void store_sse<float>(void*, void*);
	template void store_sse<double>(void*, void*);
}
