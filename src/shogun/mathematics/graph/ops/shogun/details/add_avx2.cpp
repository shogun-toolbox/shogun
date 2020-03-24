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
	void add_kernel_implementation_avx2(
	    void* input1, void* input2, void* output);

	template <typename T>
	void add_kernel_implementation_avx2(
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
	void add_kernel_implementation_avx2<float>(
		void* input1, void* input2, void* output)
	{
		static_cast<Packet*>(output)->m_data = padd(
			std::get<Packet8f>(static_cast<const Packet*>(input1)->m_data), 
			std::get<Packet8f>(static_cast<const Packet*>(input2)->m_data));
	}

	template<>
	void add_kernel_implementation_avx2<double>(
		void* input1, void* input2, void* output)
	{
		static_cast<Packet*>(output)->m_data = padd(
			std::get<Packet4d>(static_cast<const Packet*>(input1)->m_data), 
			std::get<Packet4d>(static_cast<const Packet*>(input2)->m_data));
	}

	template<>
	void add_kernel_implementation_avx2<int32_t>(
		void* input1, void* input2, void* output)
	{
		auto p1 = std::get<Packet8i>(static_cast<const Packet*>(input1)->m_data);
		auto p2 = std::get<Packet8i>(static_cast<const Packet*>(input2)->m_data);
		static_cast<Packet*>(output)->m_data = _mm256_add_epi32(p1, p2);
	}

	template void add_kernel_implementation_avx2<bool>(void*, void*, void*);
	template void add_kernel_implementation_avx2<int8_t>(void*, void*, void*);
	template void add_kernel_implementation_avx2<int16_t>(void*, void*, void*);
	template void add_kernel_implementation_avx2<int32_t>(void*, void*, void*);
	template void add_kernel_implementation_avx2<int64_t>(void*, void*, void*);
	template void add_kernel_implementation_avx2<uint8_t>(void*, void*, void*);
	template void add_kernel_implementation_avx2<uint16_t>(void*, void*, void*);
	template void add_kernel_implementation_avx2<uint32_t>(void*, void*, void*);
	template void add_kernel_implementation_avx2<uint64_t>(void*, void*, void*);
	template void add_kernel_implementation_avx2<float>(void*, void*, void*);
	template void add_kernel_implementation_avx2<double>(void*, void*, void*);
}