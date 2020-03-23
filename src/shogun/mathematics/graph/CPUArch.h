/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_GRAPH_CPU_H_
#define SHOGUN_GRAPH_CPU_H_

#include <type_traits>
#include <shogun/mathematics/graph/shogun-engine_export.h>

namespace shogun
{
	namespace graph
	{
		class SHOGUN_ENGINE_EXPORT CPUArch
		{
			enum class SIMD
			{
				NONE = 0,
				SSE = 1u << 0,
				SSE2 = 1u << 1,
				SSE3 = 1u << 2,
				SSSE3 = 1u << 3,
				SSE4_1 = 1u << 4,
				SSE4_2 = 1u << 5,
				AVX = 1u << 6,
				AVX2 = 1u << 7,
				AVX512F = 1u << 8 // TODO add all possible AVX512 instructions
				                  // -> AVX512VL, AVX512BW, etc..
			};

			inline SIMD& enable_simd(SIMD& lhs, const SIMD rhs)
			{
				using underlying = typename std::underlying_type<SIMD>::type;
				lhs = static_cast<SIMD>(
				    static_cast<underlying>(lhs) |
				    static_cast<underlying>(rhs));
				return lhs;
			}

			bool has(const SIMD instruction) const noexcept
			{
				using underlying = typename std::underlying_type<SIMD>::type;
				return static_cast<bool>(
				    static_cast<underlying>(instructions) &
				    static_cast<underlying>(instruction));
			}

			CPUArch();

			SIMD instructions;

		public:
			~CPUArch() = default;

			static CPUArch* instance();

			const bool has_sse() const noexcept
			{
				return has(SIMD::SSE);
			}

			const bool has_sse2() const noexcept
			{
				return has(SIMD::SSE2);
			}

			const bool has_sse3() const noexcept
			{
				return has(SIMD::SSE3);
			}

			const bool has_ssse3() const noexcept
			{
				return has(SIMD::SSSE3);
			}

			const bool has_sse4_1() const noexcept
			{
				return has(SIMD::SSE4_1);
			}

			const bool has_sse4_2() const noexcept
			{
				return has(SIMD::SSE4_2);
			}

			const bool has_avx() const noexcept
			{
				return has(SIMD::AVX);
			}

			const bool has_avx2() const noexcept
			{
				return has(SIMD::AVX2);
			}

			const bool has_avx512f() const noexcept
			{
				return has(SIMD::AVX512F);
			}
		};
	}
}

#endif