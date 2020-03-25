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
	void add_kernel_implementation_sse2(
	    void* input1, void* input2, void* output);

	template <typename T>
	void add_kernel_implementation_sse2(
	    void* input1, void* input2, void* output)
	{
		using vector_type = typename alignedvector_from_builtintype<T, SSE_BYTESIZE>::type;
		const auto& vec1 = std::get<vector_type>(static_cast<const Packet*>(input1)->m_data);
		const auto& vec2 = std::get<vector_type>(static_cast<const Packet*>(input2)->m_data);
		vector_type result;

		std::transform(
		    reinterpret_cast<const T*>(&vec1),
		    reinterpret_cast<const T*>(&vec1) + SSE_BYTESIZE/sizeof(T),
		    reinterpret_cast<const T*>(&vec2),
		    reinterpret_cast<T*>(&result),
		    std::plus<T>());

		static_cast<Packet*>(output)->m_data = result;
	}

	template <>
	void add_kernel_implementation_sse2<bool>(
	    void* input1, void* input2, void* output)
	{
		const auto& vec1 = std::get<bool*>(static_cast<const Packet*>(input1)->m_data);
		const auto& vec2 = std::get<bool*>(static_cast<const Packet*>(input2)->m_data);
		auto& out = std::get<bool*>(static_cast<const Packet*>(output)->m_data);

		std::transform(vec1, vec1 + SSE_BYTESIZE/sizeof(bool), vec2, out, std::plus<bool>());
	}

	template<>
	void add_kernel_implementation_sse2<float>(
		void* input1, void* input2, void* output)
	{
		static_cast<Packet*>(output)->m_data = padd(
			std::get<Packet4f>(static_cast<const Packet*>(input1)->m_data), 
			std::get<Packet4f>(static_cast<const Packet*>(input2)->m_data));
	}

	template<>
	void add_kernel_implementation_sse2<double>(
		void* input1, void* input2, void* output)
	{
		static_cast<Packet*>(output)->m_data = padd(
			std::get<Packet2d>(static_cast<const Packet*>(input1)->m_data), 
			std::get<Packet2d>(static_cast<const Packet*>(input2)->m_data));
	}

	template<>
	void add_kernel_implementation_sse2<int8_t>(
		void* input1, void* input2, void* output)
	{
		auto p1 = std::get<Packet4i>(static_cast<const Packet*>(input1)->m_data);
		auto p2 = std::get<Packet4i>(static_cast<const Packet*>(input2)->m_data);
		static_cast<Packet*>(output)->m_data = _mm_add_epi8(p1, p2);
	}

	template<>
	void add_kernel_implementation_sse2<int16_t>(
		void* input1, void* input2, void* output)
	{
		auto p1 = std::get<Packet4i>(static_cast<const Packet*>(input1)->m_data);
		auto p2 = std::get<Packet4i>(static_cast<const Packet*>(input2)->m_data);
		static_cast<Packet*>(output)->m_data = _mm_add_epi16(p1, p2);
	}

	template<>
	void add_kernel_implementation_sse2<int32_t>(
		void* input1, void* input2, void* output)
	{
		auto p1 = std::get<Packet4i>(static_cast<const Packet*>(input1)->m_data);
		auto p2 = std::get<Packet4i>(static_cast<const Packet*>(input2)->m_data);
		static_cast<Packet*>(output)->m_data = _mm_add_epi32(p1, p2);
	}

	template<>
	void add_kernel_implementation_sse2<int64_t>(
		void* input1, void* input2, void* output)
	{
		auto p1 = std::get<Packet4i>(static_cast<const Packet*>(input1)->m_data);
		auto p2 = std::get<Packet4i>(static_cast<const Packet*>(input2)->m_data);
		static_cast<Packet*>(output)->m_data = _mm_add_epi64(p1, p2);
	}

	template void add_kernel_implementation_sse2<bool>(void*, void*, void*);
	template void add_kernel_implementation_sse2<int8_t>(void*, void*, void*);
	template void add_kernel_implementation_sse2<int16_t>(void*, void*, void*);
	template void add_kernel_implementation_sse2<int32_t>(void*, void*, void*);
	template void add_kernel_implementation_sse2<int64_t>(void*, void*, void*);
	template void add_kernel_implementation_sse2<uint8_t>(void*, void*, void*);
	template void add_kernel_implementation_sse2<uint16_t>(void*, void*, void*);
	template void add_kernel_implementation_sse2<uint32_t>(void*, void*, void*);
	template void add_kernel_implementation_sse2<uint64_t>(void*, void*, void*);
	template void add_kernel_implementation_sse2<float>(void*, void*, void*);
	template void add_kernel_implementation_sse2<double>(void*, void*, void*);
}