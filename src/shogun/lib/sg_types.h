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
		using Head = None;
		static constexpr int size = 0;
	};

	template <typename T1, typename... Args>
	struct Types<T1, Args...> : Types<Args...>
	{
		using Tail = Types<Args...>;
		using Head = T1;
		static constexpr int size = sizeof...(Args) + 1;
	};

	using sg_feature_types = Types<
	    Unknown, bool, char, uint8_t, int16_t, uint16_t, int32_t, uint32_t,
	    int64_t, uint64_t, float32_t, float64_t, floatmax_t, Any>;

	using sg_all_primitive_types = Types<
	    int8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t,
	    float32_t, float64_t, floatmax_t, complex128_t, char, bool>;

	using sg_non_complex_types = Types<
	    int8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t,
	    float32_t, float64_t, floatmax_t>;

	using sg_real_types = Types<float32_t, float64_t, floatmax_t>;

	using sg_non_integer_types =
	    Types<float32_t, float64_t, floatmax_t, complex128_t>;

	using sg_numeric_types = Types<
	    int8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t,
	    float32_t, float64_t, floatmax_t, complex128_t>;

	namespace types_detail
	{
		template <typename T1, int index, bool>
		struct getTypeFromIndexImpl;

		template <typename T1, int index>
		struct getTypeFromIndexImpl<T1, index, false>
		{
			using type = None;
		};

		template <typename T1, int index>
		struct getTypeFromIndexImpl<T1, index, true>
		{
			using type = std::conditional_t<
			    (T1::size == index), typename T1::Head,
			    typename getTypeFromIndexImpl<
			        typename T1::Tail, index, (T1::size > 1)>::type>;
		};

		template <typename TypesT, typename T>
		struct appendToTypesImpl;

		template <
		    template <typename...> class TypesT, typename T, typename... Args>
		struct appendToTypesImpl<TypesT<Args...>, T>
		{
			using type = TypesT<T, Args...>;
		};

		template <typename TypesT1, typename TypesT2>
		struct appendTypesImpl;

		template <
		    template <typename...> class TypesT1,
		    template <typename...> class TypesT2, typename... Args1,
		    typename... Args2>
		struct appendTypesImpl<TypesT1<Args1...>, TypesT2<Args2...>>
		{
			static_assert(
			    std::is_same<TypesT1<>, TypesT2<>>::value,
			    "Expected types to be the same");
			using type = TypesT1<Args2..., Args1...>;
		};
	} // namespace types_detail

	template <typename TypesT, int index>
	struct getTypeFromIndex
	{
		static_assert(
		    index >= 0, "Index must be greater than or equal to zero");
		static_assert(index < TypesT::size, "Index is out of bounds");
		using type = typename types_detail::getTypeFromIndexImpl<
		    TypesT, TypesT::size - index, true>::type;
	};

	template <typename TypesT, typename T1>
	struct appendToTypes
	{
		using type = typename types_detail::appendToTypesImpl<TypesT, T1>::type;
	};

	template <typename TypesT1, typename TypesT2>
	struct appendTypes
	{
		using type =
		    typename types_detail::appendTypesImpl<TypesT1, TypesT2>::type;
	};

	namespace types_detail
	{

		template <typename TypesT1, typename TypesT2, bool>
		struct reverseTypesImpl;

		template <typename TypesT1, typename TypesT2>
		struct reverseTypesImpl<TypesT1, TypesT2, false>
		{
			using type = None;
		};

		template <typename TypesT1, typename TypesT2>
		struct reverseTypesImpl<TypesT1, TypesT2, true>
		{
			using type = std::conditional_t<
			    (TypesT2::size == 1),
			    typename appendToTypes<TypesT1, typename TypesT2::Head>::type,
			    typename reverseTypesImpl<
			        typename appendToTypes<
			            TypesT1, typename TypesT2::Head>::type,
			        typename TypesT2::Tail, (TypesT2::size > 1)>::type>;
		};

	} // namespace types_detail

	template <typename TypesT>
	struct reverseTypes
	{
		using type = typename types_detail::reverseTypesImpl<
		    Types<>, TypesT, true>::type;
	};

	namespace types_detail
	{

		template <typename TypesTHead, typename TypesTTail, typename, bool>
		struct popTypesImplType_;

		template <typename TypesTHead, typename TypesTTail, typename T1>
		struct popTypesImplType_<TypesTHead, TypesTTail, T1, false>
		{
			using type = typename TypesTTail::Tail;
		};

		template <typename TypesTHead, typename TypesTTail, typename T1>
		struct popTypesImplType_<TypesTHead, TypesTTail, T1, true>
		{
			using type = typename appendTypes<
			    typename TypesTTail::Tail,
			    typename reverseTypes<TypesTHead>::type>::type;
		};

		template <typename TypesTHead, typename TypesTTail, typename T, bool>
		struct popTypesImplType;

		template <typename TypesTHead, typename TypesTTail, typename T>
		struct popTypesImplType<TypesTHead, TypesTTail, T, false>
		{
			using type = None;
		};

		template <typename TypesTHead, typename TypesTTail, typename T>
		struct popTypesImplType<TypesTHead, TypesTTail, T, true>
		{
			using type = std::conditional_t<
			    std::is_same<typename TypesTTail::Head, T>::value,
			    typename popTypesImplType_<
			        TypesTHead, TypesTTail, T, (TypesTHead::size > 0)>::type,
			    typename popTypesImplType<
			        typename appendToTypes<
			            TypesTHead, typename TypesTTail::Head>::type,
			        typename TypesTTail::Tail, T,
			        (TypesTTail::size > 1)>::type>;
		};
	} // namespace types_detail

	template <typename TypesT, typename T1>
	struct popTypesByType
	{
		using type = typename types_detail::popTypesImplType<
		    Types<>, TypesT, T1, true>::type;
	};

	namespace types_detail
	{

		template <typename TypesT, typename T1, bool>
		struct popTypesImplTypes;

		template <typename TypesT, typename T1>
		struct popTypesImplTypes<TypesT, T1, false>
		{
			using type = None;
		};

		template <typename TypesT, typename T1>
		struct popTypesImplTypes<TypesT, T1, true>
		{
			using type = std::conditional_t<
			    (T1::size == 1),
			    typename popTypesByType<TypesT, typename T1::Head>::type,
			    typename popTypesImplTypes<
			        typename popTypesByType<TypesT, typename T1::Head>::type,
			        typename T1::Tail, (T1::size > 1)>::type>;
		};
	} // namespace types_detail

	template <typename TypesT, typename TypesT1>
	struct popTypesByTypes
	{
		using type = typename types_detail::popTypesImplTypes<
		    TypesT, TypesT1, true>::type;
	};
#endif // DOXYGEN_SHOULD_SKIP_THIS
} // namespace shogun

#endif // SHOGUN_TYPE_H