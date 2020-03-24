/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_PACKET_SHOGUN_H_
#define SHOGUN_PACKET_SHOGUN_H_

#include "details/PacketType.h"

namespace shogun
{
	namespace graph
	{
		namespace op
		{
			// template <typename T>
			// aligned_vector load_sse(void* input1);

			template <typename T>
			aligned_vector load_avx(void* input1);

			// template <typename T>
			// aligned_vector load_avx512f(void* input1);

			// template <typename T>
			// void store_sse(const aligned_vector& input1, void* output);

			template <typename T>
			void store_avx(const aligned_vector& input1, void* output);

			// template <typename T>
			// void store_avx512f(const aligned_vector& input1, void* output);

			enum class RegisterType
			{
				SSE = 16,
				AVX = 32,
				AVX512 = 64
			};

			struct Packet
			{
				Packet() = delete;

				template <typename T>
				Packet(const T* data, const RegisterType register_type): m_register_type(register_type)
				{
					switch(register_type)
					{
						// case RegisterType::SSE:
						// 	m_data = load_sse<T>((void*)data);
						// 	break;
						case RegisterType::AVX:
							m_data = load_avx<T>((void*)data);
							break;
						// case RegisterType::AVX512:
						// 	m_data = load_avx512f<T>((void*)data);
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
						// case RegisterType::SSE:
						// 	store_sse<T>(m_data, output);
						// 	break;
						case RegisterType::AVX:
							store_avx<T>(m_data, output);
							break;
						// case RegisterType::AVX512:
						// 	store_avx512f<T>(m_data, output);
					}
				}

				const size_t byte_size() const noexcept
				{
					return static_cast<size_t>(m_register_type);
				}
				
				aligned_vector m_data;
				const RegisterType m_register_type;
			};
		}
	}
}

#endif