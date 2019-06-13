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

		template<typename T, typename _ = void>
		struct is_pair : std::false_type {};

		template<typename T>
		struct is_pair<
			T,
			std::conditional_t<
				false,
				try_to_declare<
					decltype(std::declval<T>().first),
					decltype(std::declval<T>().second)
				>,
				void
			>
		> : public std::true_type {};

		template<typename T, typename _ = void>
		struct has_equals : std::false_type {};

		template<typename T>
		struct has_equals<
			T,
			std::conditional_t<
				false,
				try_to_declare<
					decltype(std::declval<T>().equals(std::declval<T>()))
				>,
				void
			>
		> : public std::true_type {};

		template<typename T, typename _ = void>
		struct has_equals_ptr : std::false_type {};

		template<typename T>
		struct has_equals_ptr<
			T,
			std::conditional_t<
				false,
				try_to_declare<
					decltype(std::declval<T>()->equals(std::declval<T>()))
				>,
				void
			>
		> : public std::true_type {};

		template<typename T, typename _ = void>
		struct is_comparable : std::false_type {};

		template<typename T>
		struct is_comparable<
			T,
			std::conditional_t<
				false,
				try_to_declare<
					decltype(std::declval<T>() == std::declval<T>())
				>,
				void
			>
		> : public std::true_type {};

	} // namespace traits
} // namespace shogun
#endif
