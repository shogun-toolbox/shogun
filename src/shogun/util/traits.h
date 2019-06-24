#ifndef __UTIL_TRAITS_H__
#define __UTIL_TRAITS_H__

#include <type_traits>

namespace shogun
{
	namespace traits
	{
		template<typename... Ts>
		struct try_to_declare {};

		template<typename... Ts>
		using when_exists = std::conditional_t<false, try_to_declare<Ts...>, void>;

		template<typename T, typename _ = void>
		struct is_container : std::false_type {};

		template<typename T>
		struct is_container<
			T,
			when_exists<
				typename T::value_type,
				typename T::size_type,
				typename T::allocator_type,
				typename T::iterator,
				typename T::const_iterator,
				decltype(std::declval<T>().size()),
				decltype(std::declval<T>().begin()),
				decltype(std::declval<T>().end()),
				decltype(std::declval<T>().cbegin()),
				decltype(std::declval<T>().cend())
			>
		> : public std::true_type {};

		template<typename T, typename _ = void>
		struct is_hashable : std::false_type {};

		template<typename T>
		struct is_hashable<
			T,
			when_exists<
				decltype(std::hash<T>{}(std::declval<T>()))
			>
		> : public std::true_type {};

		template<typename T, typename _ = void>
		struct is_pair : std::false_type {};

		template<typename T>
		struct is_pair<
			T,
			when_exists<
				decltype(std::declval<T>().first),
				decltype(std::declval<T>().second)
			>
		> : public std::true_type {};

		template<typename T, typename _ = void>
		struct has_equals : std::false_type {};

		template<typename T>
		struct has_equals<
			T,
			when_exists<
				decltype(std::declval<T>().equals(std::declval<T>()))
			>
		> : public std::true_type {};

		template<typename T, typename _ = void>
		struct has_equals_ptr : std::false_type {};

		template<typename T>
		struct has_equals_ptr<
			T,
			when_exists<
				decltype(std::declval<T>()->equals(std::declval<T>()))
			>
		> : public std::true_type {};

		template<typename T, typename _ = void>
		struct is_comparable : std::false_type {};

		template<typename T>
		struct is_comparable<
			T,
			when_exists<
				decltype(std::declval<T>() == std::declval<T>())
			>
		> : public std::true_type {};

		template<typename T, typename _ = void>
		struct is_functional : std::false_type {};

		template<typename T>
		struct is_functional<
			T,
			when_exists<
				decltype(std::declval<T>()()),
				typename T::result_type
			>
		> : public std::true_type {};

		template<typename T, typename _ = void>
		struct is_visitable : public std::conditional_t<is_functional<T>::value, std::false_type, std::true_type> {};

		template<typename T>
		struct is_visitable<T, when_exists<typename T::not_visitable>> : public std::false_type {};

		template<typename T>
		using returns_void = std::is_same<typename T::result_type, void>;

	} // namespace traits
} // namespace shogun
#endif

