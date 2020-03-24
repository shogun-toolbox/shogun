/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include <Eigen/Core>

using namespace Eigen::internal;

namespace shogun::graph::op {
	template <typename T>
	void load_avx512f(void* input, void*& output);

	template <typename T>
	void load_avx512f(void* data, void*& output)
	{
		output = data;
	}

	template <>
	void load_avx512f<float>(void* data, void*& output)
	{
		auto val = ploadu<Packet16f>(static_cast<const float*>(data));
		output = (void*)&val;
	}

	template <>
	void load_avx512f<double>(void* data, void*& output)
	{
		auto val = ploadu<Packet8d>(static_cast<const double*>(data));
		output = (void*)&val;
	}

	template <>
	void load_avx512f<int>(void* data, void*& output)
	{
		auto val = ploadu<Packet16i>(static_cast<const int*>(data));
		output = (void*)&val;
	}

	template void load_avx512f<bool>(void*, void*&);
	template void load_avx512f<int8_t>(void*, void*&);
	template void load_avx512f<int16_t>(void*, void*&);
	template void load_avx512f<int32_t>(void*, void*&);
	template void load_avx512f<int64_t>(void*, void*&);
	template void load_avx512f<uint8_t>(void*, void*&);
	template void load_avx512f<uint16_t>(void*, void*&);
	template void load_avx512f<uint32_t>(void*, void*&);
	template void load_avx512f<uint64_t>(void*, void*&);
	template void load_avx512f<float>(void*, void*&);
	template void load_avx512f<double>(void*, void*&);
}
