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
				__m512, __m512d, __m512i, bool*, int8_t*, uint8_t*, int16_t*, uint16_t*, int32_t*,
				uint32_t*, int64_t*, uint64_t*, float*, double*, std::nullptr_t>;
		}
	}
}		

#endif