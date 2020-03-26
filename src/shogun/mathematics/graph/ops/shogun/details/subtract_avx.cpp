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

	template <typename T>
	void subtract_kernel_implementation_avx(
	    void* input1, void* input2, void* output);

	template <typename T>
	void subtract_kernel_implementation_avx(
	    void* input1, void* input2, void* output)
	{
		using vector_type = typename alignedvector_from_builtintype<T, AVX_BYTESIZE>::type;
		const auto& vec1 = std::get<vector_type>(static_cast<const Packet*>(input1)->m_data);
		const auto& vec2 = std::get<vector_type>(static_cast<const Packet*>(input2)->m_data);
		vector_type result;

		std::transform(
		    reinterpret_cast<const T*>(&vec1),
		    reinterpret_cast<const T*>(&vec1) + AVX_BYTESIZE/sizeof(T),
		    reinterpret_cast<const T*>(&vec2),
		    reinterpret_cast<T*>(&result),
		    std::minus<T>());

		static_cast<Packet*>(output)->m_data = result;
	}

	template<>
	void subtract_kernel_implementation_avx<float>(
		void* input1, void* input2, void* output)
	{
		static_cast<Packet*>(output)->m_data = psub(
			std::get<Packet8f>(static_cast<const Packet*>(input1)->m_data), 
			std::get<Packet8f>(static_cast<const Packet*>(input2)->m_data));
	}

	template<>
	void subtract_kernel_implementation_avx<double>(
		void* input1, void* input2, void* output)
	{
		static_cast<Packet*>(output)->m_data = psub(
			std::get<Packet4d>(static_cast<const Packet*>(input1)->m_data), 
			std::get<Packet4d>(static_cast<const Packet*>(input2)->m_data));
	}

	template void subtract_kernel_implementation_avx<bool>(void*, void*, void*);
	template void subtract_kernel_implementation_avx<int8_t>(void*, void*, void*);
	template void subtract_kernel_implementation_avx<int16_t>(void*, void*, void*);
	template void subtract_kernel_implementation_avx<int32_t>(void*, void*, void*);
	template void subtract_kernel_implementation_avx<int64_t>(void*, void*, void*);
	template void subtract_kernel_implementation_avx<uint8_t>(void*, void*, void*);
	template void subtract_kernel_implementation_avx<uint16_t>(void*, void*, void*);
	template void subtract_kernel_implementation_avx<uint32_t>(void*, void*, void*);
	template void subtract_kernel_implementation_avx<uint64_t>(void*, void*, void*);
	template void subtract_kernel_implementation_avx<float>(void*, void*, void*);
	template void subtract_kernel_implementation_avx<double>(void*, void*, void*);
}