#ifndef __UTIL_TRAITS_H__
#define __UTIL_TRAITS_H__

#include <type_traits>

namespace shogun
{
	namespace utils
	{
		template<typename T, typename _ = void>
		struct is_container : std::false_type {};

		template<typename... Ts>
		struct is_container_helper {};

		template<typename T>
		struct is_container<
			T,
			std::conditional_t<
				false,
				is_container_helper<
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
	} // namespace utils
} // namespace shogun
#endif
