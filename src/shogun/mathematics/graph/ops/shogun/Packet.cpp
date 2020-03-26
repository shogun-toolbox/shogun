/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include "Packet.h"

#include <shogun/io/SGIO.h>

using namespace shogun;
using namespace shogun::graph::op;

template <typename T>
Packet::Packet(const T* data, const RegisterType register_type): m_register_type(register_type)
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
			break;
		case RegisterType::SCALAR:
			m_data = *data;
	}
}

Packet::Packet(const RegisterType register_type): m_data(nullptr), m_register_type(register_type)
{
}

template <typename T>
void Packet::store(T* output)
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
			break;
		case RegisterType::SCALAR:
		{
			// copies value from stack to array
			*output = std::get<T>(m_data);
		}
	}
}

const size_t Packet::byte_size() const noexcept
{
	return static_cast<size_t>(m_register_type);
}

template void Packet::store(bool*);
template void Packet::store(int8_t*);
template void Packet::store(int16_t*);
template void Packet::store(int32_t*);
template void Packet::store(int64_t*);
template void Packet::store(uint8_t*);
template void Packet::store(uint16_t*);
template void Packet::store(uint32_t*);
template void Packet::store(uint64_t*);
template void Packet::store(float*);
template void Packet::store(double*);

template Packet::Packet(const bool*, const RegisterType);
template Packet::Packet(const int8_t*, const RegisterType);
template Packet::Packet(const int16_t*, const RegisterType);
template Packet::Packet(const int32_t*, const RegisterType);
template Packet::Packet(const int64_t*, const RegisterType);
template Packet::Packet(const uint8_t*, const RegisterType);
template Packet::Packet(const uint16_t*, const RegisterType);
template Packet::Packet(const uint32_t*, const RegisterType);
template Packet::Packet(const uint64_t*, const RegisterType);
template Packet::Packet(const float*, const RegisterType);
template Packet::Packet(const double*, const RegisterType);