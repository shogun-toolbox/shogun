/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include <algorithm>
#include <functional>
#include <Eigen/Core>
#include "PacketType.h"

using namespace Eigen::internal;

namespace shogun::graph::op {

	struct Packet
	{
		aligned_vector m_data;
		const size_t m_size;
	};

	template <typename T>
	void add_kernel_implementation_avx(
	    void* input1, void* input2, void* output);

	template <typename T>
	void add_kernel_implementation_avx(
	    void* input1, void* input2, void* output)
	{
		std::transform(
		    std::get<T*>(static_cast<const Packet*>(input1)->m_data),
		    std::get<T*>(static_cast<const Packet*>(input1)->m_data) + 32/sizeof(T),
		    std::get<T*>(static_cast<const Packet*>(input2)->m_data), 
		    std::get<T*>(static_cast<const Packet*>(output)->m_data),
		    std::plus<T>());
	}

	template<>
	void add_kernel_implementation_avx<float>(
		void* input1, void* input2, void* output)
	{
		static_cast<Packet*>(output)->m_data = padd(
			std::get<Packet8f>(static_cast<const Packet*>(input1)->m_data), 
			std::get<Packet8f>(static_cast<const Packet*>(input2)->m_data));
	}

	template<>
	void add_kernel_implementation_avx<double>(
		void* input1, void* input2, void* output)
	{
		static_cast<Packet*>(output)->m_data = padd(
			std::get<Packet4d>(static_cast<const Packet*>(input1)->m_data), 
			std::get<Packet4d>(static_cast<const Packet*>(input2)->m_data));
	}

	template void add_kernel_implementation_avx<bool>(void*, void*, void*);
	template void add_kernel_implementation_avx<int8_t>(void*, void*, void*);
	template void add_kernel_implementation_avx<int16_t>(void*, void*, void*);
	template void add_kernel_implementation_avx<int32_t>(void*, void*, void*);
	template void add_kernel_implementation_avx<int64_t>(void*, void*, void*);
	template void add_kernel_implementation_avx<uint8_t>(void*, void*, void*);
	template void add_kernel_implementation_avx<uint16_t>(void*, void*, void*);
	template void add_kernel_implementation_avx<uint32_t>(void*, void*, void*);
	template void add_kernel_implementation_avx<uint64_t>(void*, void*, void*);
	template void add_kernel_implementation_avx<float>(void*, void*, void*);
	template void add_kernel_implementation_avx<double>(void*, void*, void*);
}