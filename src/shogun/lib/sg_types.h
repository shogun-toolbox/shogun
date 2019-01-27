/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_TYPE_H
#define SHOGUN_TYPE_H

#include <shogun/lib/any.h>

using namespace shogun;

namespace shogun
{
	// utility structs
	struct Unknown
	{
	};
	struct None
	{
	};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
	// struct to store types
	template <typename... Args>
	struct Types
	{
		typedef None Head;
		static constexpr int size = 0;
	};

	template <typename T1, typename... Args>
	struct Types<T1, Args...> : Types<Args...>
	{
		typedef Types<Args...> Tail;
		typedef T1 Head;
		static constexpr int size = sizeof...(Args) + 1;
	};
#endif // DOXYGEN_SHOULD_SKIP_THIS

	// Type definitions
	// NOTE: order of Types is reverse of enum as it uses last in first out
	// approach.
	typedef Types<
	    Any, floatmax_t, float64_t, float32_t, uint64_t, int64_t, uint32_t,
	    int32_t, uint16_t, int16_t, uint8_t, char, bool, Unknown>
	    sg_feature_types;

	// TODO: add bool and complex128_t types
	// TODO: in C++17 can use constexpr if (...) to check types in tests
	// 		 to skip types that we don't write tests for
	typedef Types<
	    int8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t,
	    float32_t, float64_t, floatmax_t, char>
	    sg_all_primitive_types;

	typedef Types<
	    int8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t,
	    float32_t, float64_t, floatmax_t>
	    sg_non_complex_types;

	typedef Types<float32_t, float64_t, floatmax_t> sg_real_types;

	// TODO: add complex128_t type
	typedef Types<float32_t, float64_t, floatmax_t> sg_non_integer_types;

	namespace types_detail
	{
		template <typename T1, int index, bool>
		struct getTypeFromIndex_impl
		{
		};

		template <typename T1, int index>
		struct getTypeFromIndex_impl<T1, index, false>
		{
			using type = None;
		};

		template <typename T1, int index>
		struct getTypeFromIndex_impl<T1, index, true>
		{
			using type = std::conditional_t<
			    (T1::size == index),
			    typename T1::Head,
			    typename getTypeFromIndex_impl<
			        typename T1::Tail, index, (T1::size > 1)>::type>;
		};
	} // namespace types_detail

	template <typename T1, int index>
	struct getTypeFromIndex
	{
		using type = std::conditional_t<
		    (index >= 0) && (index < T1::size),
		    typename types_detail::getTypeFromIndex_impl<T1, index+1, true>::type,
		    None>;
	};
} // namespace shogun

#endif // SHOGUN_TYPE_H