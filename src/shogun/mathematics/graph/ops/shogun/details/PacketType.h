/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_PACKET_TYPES_SHOGUN_H_
#define SHOGUN_PACKET_TYPES_SHOGUN_H_

#include <cstdint>
#include <immintrin.h>
#include <variant>

namespace shogun
{
	namespace graph
	{
		namespace op
		{
			using aligned_vector = std::variant<__m128, __m128d, __m128i, __m256, __m256d, __m256i, 
				__m512, __m512d, __m512i, bool*, std::nullptr_t>;

			static constexpr size_t SSE_BYTESIZE    = 16;
			static constexpr size_t AVX_BYTESIZE    = 32;
			static constexpr size_t AVX512_BYTESIZE = 64;

			template <typename T, size_t register_size>
			struct alignedvector_from_builtintype
			{};

			template <>
			struct alignedvector_from_builtintype<int8_t, 16>
			{
				using type = __m128i;
			};

			template <>
			struct alignedvector_from_builtintype<uint8_t, 16>
			{
				using type = __m128i;
			};

			template <>
			struct alignedvector_from_builtintype<int16_t, 16>
			{
				using type = __m128i;
			};

			template <>
			struct alignedvector_from_builtintype<uint16_t, 16>
			{
				using type = __m128i;
			};

			template <>
			struct alignedvector_from_builtintype<int32_t, 16>
			{
				using type = __m128i;
			};

			template <>
			struct alignedvector_from_builtintype<uint32_t, 16>
			{
				using type = __m128i;
			};

			template <>
			struct alignedvector_from_builtintype<int64_t, 16>
			{
				using type = __m128i;
			};

			template <>
			struct alignedvector_from_builtintype<uint64_t, 16>
			{
				using type = __m128i;
			};

			template <>
			struct alignedvector_from_builtintype<float, 16>
			{
				using type = __m128;
			};

			template <>
			struct alignedvector_from_builtintype<double, 16>
			{
				using type = __m128d;
			};
		
			//=====================================================
			template <>
			struct alignedvector_from_builtintype<int8_t, 32>
			{
				using type = __m256i;
			};

			template <>
			struct alignedvector_from_builtintype<uint8_t, 32>
			{
				using type = __m256i;
			};

			template <>
			struct alignedvector_from_builtintype<int16_t, 32>
			{
				using type = __m256i;
			};

			template <>
			struct alignedvector_from_builtintype<uint16_t, 32>
			{
				using type = __m256i;
			};

			template <>
			struct alignedvector_from_builtintype<int32_t, 32>
			{
				using type = __m256i;
			};

			template <>
			struct alignedvector_from_builtintype<uint32_t, 32>
			{
				using type = __m256i;
			};

			template <>
			struct alignedvector_from_builtintype<int64_t, 32>
			{
				using type = __m256i;
			};

			template <>
			struct alignedvector_from_builtintype<uint64_t, 32>
			{
				using type = __m256i;
			};

			template <>
			struct alignedvector_from_builtintype<float, 32>
			{
				using type = __m256;
			};

			template <>
			struct alignedvector_from_builtintype<double, 32>
			{
				using type = __m256d;
			};
			//==================================================
			template <>
			struct alignedvector_from_builtintype<int8_t, 64>
			{
				using type = __m512i;
			};

			template <>
			struct alignedvector_from_builtintype<uint8_t, 64>
			{
				using type = __m512i;
			};

			template <>
			struct alignedvector_from_builtintype<int16_t, 64>
			{
				using type = __m512i;
			};

			template <>
			struct alignedvector_from_builtintype<uint16_t, 64>
			{
				using type = __m512i;
			};

			template <>
			struct alignedvector_from_builtintype<int32_t, 64>
			{
				using type = __m512i;
			};

			template <>
			struct alignedvector_from_builtintype<uint32_t, 64>
			{
				using type = __m512i;
			};

			template <>
			struct alignedvector_from_builtintype<int64_t, 64>
			{
				using type = __m512i;
			};

			template <>
			struct alignedvector_from_builtintype<uint64_t, 64>
			{
				using type = __m512i;
			};

			template <>
			struct alignedvector_from_builtintype<float, 64>
			{
				using type = __m512;
			};

			template <>
			struct alignedvector_from_builtintype<double, 64>
			{
				using type = __m512d;
			};
		}
	}
}		

#endif