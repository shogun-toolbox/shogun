/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include <Eigen/Core>

using namespace Eigen::internal;

namespace shogun::graph::op {
	template <typename T>
	void load_sse(void* input, void*& output);

	template <typename T>
	void load_sse(void* data, void*& output)
	{
		output = data;
	}

	template <>
	void load_sse<float>(void* data, void*& output)
	{
		auto val = ploadu<Packet4f>(static_cast<const float*>(data));
		output = (void*)&val;
	}

	template <>
	void load_sse<double>(void* data, void*& output)
	{
		auto val = ploadu<Packet2d>(static_cast<const double*>(data));
		output = (void*)&val;
	}

	template <>
	void load_sse<int>(void* data, void*& output)
	{
		auto val = ploadu<Packet4i>(static_cast<const int*>(data));
		output = (void*)&val;
	}

	template void load_sse<bool>(void*, void*&);
	template void load_sse<int8_t>(void*, void*&);
	template void load_sse<int16_t>(void*, void*&);
	template void load_sse<int32_t>(void*, void*&);
	template void load_sse<int64_t>(void*, void*&);
	template void load_sse<uint8_t>(void*, void*&);
	template void load_sse<uint16_t>(void*, void*&);
	template void load_sse<uint32_t>(void*, void*&);
	template void load_sse<uint64_t>(void*, void*&);
	template void load_sse<float>(void*, void*&);
	template void load_sse<double>(void*, void*&);
}
