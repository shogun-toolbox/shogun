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
	typedef Types<
	    Any, floatmax_t, float64_t, float32_t, uint64_t, int64_t, uint32_t,
	    int32_t, uint16_t, int16_t, uint8_t, char, bool, Unknown>
	    sg_feature_types;

	typedef Types<
	    int8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t,
	    float32_t, float64_t, floatmax_t, complex128_t, char, bool>
	    sg_all_primitive_types;

	typedef Types<
	    int8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t,
	    float32_t, float64_t, floatmax_t>
	    sg_non_complex_types;

	typedef Types<float32_t, float64_t, floatmax_t> sg_real_types;

	typedef Types<float32_t, float64_t, floatmax_t, complex128_t> sg_non_integer_types;

    typedef Types<int8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t,
    float32_t, float64_t, floatmax_t, complex128_t> sg_numeric_types;

    typedef Types<uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t,
            uint64_t, float32_t, float64_t, floatmax_t, char> sg_tb_types;

	namespace types_detail
	{
        template <typename T1, int index, bool>
        struct getTypeFromIndex_impl;

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
                    typename getTypeFromIndex_impl<typename T1::Tail, index, (T1::size > 1)>::type>;
        };

		template <typename TypesT, typename T>      
		struct appendToTypes_impl;

		template <template<typename...> class TypesT, typename T, typename... Args>              
		struct appendToTypes_impl <TypesT<Args...>, T >
		{                                                                               
		    using type = TypesT<T, Args...>;                                                  
		};

		template <typename TypesT1, typename TypesT2>
		struct appendTypes_impl;

		template <template<typename...> class TypesT1, template<typename...> class TypesT2, typename... Args1, typename... Args2>
		struct appendTypes_impl<TypesT1<Args1...>, TypesT2<Args2...>>
		{
			static_assert(std::is_same<TypesT1<>, TypesT2<>>::value, "Expected types to be the same");
			using type = TypesT1<Args2..., Args1...>;
		};
	} // namespace types_detail

	template <typename TypesT, int index>
	struct getTypeFromIndex
	{
		static_assert(index >= 0, "Index must be greater or equal than zero");
		static_assert(index < TypesT::size, "Index is out of bounds");
		using type = typename types_detail::getTypeFromIndex_impl<TypesT, TypesT::size-index, true>::type;
	};

	template <typename TypesT, typename T1>
	struct appendToTypes
	{
		using type = typename types_detail::appendToTypes_impl<TypesT, T1>::type;
	};

	template <typename TypesT1, typename TypesT2>
	struct appendTypes
	{
		using type = typename types_detail::appendTypes_impl<TypesT1, TypesT2>::type;
	};	

	namespace types_detail {

		template <typename TypesT1, typename TypesT2, bool>
		struct reverseTypes_impl;

		template <typename TypesT1, typename TypesT2>
		struct reverseTypes_impl<TypesT1, TypesT2, false>
		{
			using type = None;
		};

		template <typename TypesT1, typename TypesT2>
		struct reverseTypes_impl <TypesT1, TypesT2, true>
		{
			using type = std::conditional_t<
				(TypesT2::size == 1),
				typename appendToTypes<TypesT1, typename TypesT2::Head>::type,
				typename reverseTypes_impl<
					typename appendToTypes<
						TypesT1, typename TypesT2::Head>::type, 
					typename TypesT2::Tail, (TypesT2::size > 1)>::type>;
		};

	} // namespace types_detail

	template<typename TypesT>
	struct reverseTypes
	{
		using type = typename types_detail::reverseTypes_impl<Types<>, TypesT, true>::type;
	};

	namespace types_detail {

		template <typename TypesTHead, typename TypesTTail, typename, bool>
		struct popTypes_impl_type_;

		template <typename TypesTHead, typename TypesTTail, typename T1>
		struct popTypes_impl_type_<TypesTHead, TypesTTail, T1, false>
		{
			using type = typename TypesTTail::Tail;
		};

		template <typename TypesTHead, typename TypesTTail, typename T1>
		struct popTypes_impl_type_<TypesTHead, TypesTTail, T1, true>
		{
			using type = typename appendTypes<typename TypesTTail::Tail, typename reverseTypes<TypesTHead>::type>::type;
		};

		template <typename TypesTHead, typename TypesTTail, typename T, bool>
		struct popTypes_impl_type;

		template <typename TypesTHead, typename TypesTTail, typename T>
		struct popTypes_impl_type<TypesTHead, TypesTTail, T, false>
		{
			using type = None;
		};

		template <typename TypesTHead, typename TypesTTail, typename T>
		struct popTypes_impl_type<TypesTHead, TypesTTail, T, true>
		{
			using type = std::conditional_t<
				std::is_same<typename TypesTTail::Head, T>::value,
				typename popTypes_impl_type_<TypesTHead, TypesTTail, T, (TypesTHead::size > 0)>::type,
				typename popTypes_impl_type<typename appendToTypes<TypesTHead, typename TypesTTail::Head>::type, typename TypesTTail::Tail, T, (TypesTTail::size > 1)>::type>;
		};
	} // namespace types_detail

	template<typename TypesT, typename T1>
	struct popTypesByType
	{
		using type = typename types_detail::popTypes_impl_type<Types<>, TypesT, T1, true>::type;
	};

	namespace types_detail {

		template <typename TypesT, typename T1, bool>
		struct popTypes_impl_types;

		template <typename TypesT, typename T1>
		struct popTypes_impl_types<TypesT, T1, false>
		{
			using type = None;
		};

		template <typename TypesT, typename T1>
		struct popTypes_impl_types<TypesT, T1, true>
		{
			using type = std::conditional_t<
				(T1::size == 1), 
				typename popTypesByType<TypesT, typename T1::Head>::type,
				typename popTypes_impl_types<typename popTypesByType<TypesT, typename T1::Head>::type, typename T1::Tail, (T1::size > 1)>::type>;
		};
	} // namespace types_detail

	template<typename TypesT, typename TypesT1>
	struct popTypesByTypes
	{
		using type = typename types_detail::popTypes_impl_types<TypesT, TypesT1, true>::type;
	};
} // namespace shogun

#endif // SHOGUN_TYPE_H