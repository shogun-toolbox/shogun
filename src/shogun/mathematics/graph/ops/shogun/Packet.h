/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_PACKET_SHOGUN_H_
#define SHOGUN_PACKET_SHOGUN_H_

#include "details/PacketType.h"
#include <shogun/mathematics/graph/CPUArch.h>

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

			enum class RegisterType
			{
				// what value should a scalar have?
				SCALAR = 0,
				SSE = 16,
				AVX = 32,
				AVX512 = 64
			};

			RegisterType get_register_type_from_instructions(CPUArch::SIMD instruction)
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
			}

			struct Packet
			{
				Packet() = delete;

				template <typename T>
				Packet(const T* data, const RegisterType register_type): m_register_type(register_type)
				{
					switch(register_type)
					{
						case RegisterType::SSE:
							m_data = load_sse<T>((void*)data);
							break;
						case RegisterType::AVX:
							m_data = load_avx<T>((void*)data);
							break;
						case RegisterType::AVX512:
							m_data = load_avx512<T>((void*)data);
					}
				}

				Packet(const RegisterType register_type): m_data(nullptr), m_register_type(register_type)
				{
				}

				~Packet() = default;

				template <typename T>
				void store(T* output)
				{
					switch(m_register_type)
					{
						case RegisterType::SSE:
							store_sse<T>(m_data, output);
							break;
						case RegisterType::AVX:
							store_avx<T>(m_data, output);
							break;
						case RegisterType::AVX512:
							store_avx512<T>(m_data, output);
					}
				}

				const size_t byte_size() const noexcept
				{
					return static_cast<size_t>(m_register_type);
				}
				
				aligned_vector m_data;
				const RegisterType m_register_type;
			};

			using BinaryPacketFunction = std::function<void(const Packet&, const Packet&, const Packet&)>;
		}
	}
}

#endif