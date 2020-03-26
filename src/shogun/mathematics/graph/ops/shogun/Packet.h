/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_PACKET_SHOGUN_H_
#define SHOGUN_PACKET_SHOGUN_H_

#include "details/PacketType.h"
#include <shogun/mathematics/graph/CPUArch.h>

#include <functional>

namespace shogun
{
	namespace graph
	{
		namespace op
		{
			template <typename T>
			aligned_vector load_sse(void* input1);

			template <typename T>
			aligned_vector load_avx(void* input1);

			template <typename T>
			aligned_vector load_avx512(void* input1);

			template <typename T>
			void store_sse(const aligned_vector& input1, void* output);

			template <typename T>
			void store_avx(const aligned_vector& input1, void* output);

			template <typename T>
			void store_avx512(const aligned_vector& input1, void* output);

			inline RegisterType get_register_type_from_instructions(CPUArch::SIMD instruction)
			{
				switch(instruction)
				{
					case CPUArch::SIMD::SSE:
					case CPUArch::SIMD::SSE2:
					case CPUArch::SIMD::SSE3:
					case CPUArch::SIMD::SSSE3:
					case CPUArch::SIMD::SSE4_1:
					case CPUArch::SIMD::SSE4_2:
						return RegisterType::SSE;
					case CPUArch::SIMD::AVX:
					case CPUArch::SIMD::AVX2:
						return RegisterType::AVX;
					case CPUArch::SIMD::AVX512F:
						return RegisterType::AVX512;
					case CPUArch::SIMD::NONE:
						return RegisterType::SCALAR;
				}
				return RegisterType::SCALAR;
			}

			using BinaryPacketFunction = std::function<void(const Packet&, const Packet&, Packet&)>;
		}
	}
}

#endif