#ifndef __UTIL_TRAITS_H__
#define __UTIL_TRAITS_H__

#include <type_traits>

namespace shogun
{
	namespace traits
	{
		template<typename... Ts>
		struct try_to_declare {};

		template<typename T, typename _ = void>
		struct is_container : std::false_type {};

		template<typename T>
		struct is_container<
			T,
			std::conditional_t<
				false,
				try_to_declare<
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
					>,
				void
				>
			> : public std::true_type {};

		template<typename T, typename _ = void>
		struct is_hashable : std::false_type {};

		template<typename T>
		struct is_hashable<
			T,
			std::conditional_t<
				false,
				try_to_declare<
					decltype(std::hash<T>{}(std::declval<T>()))
				>,
				void
			>
		> : public std::true_type {};


	} // namespace utils
} // namespace shogun
#endif
