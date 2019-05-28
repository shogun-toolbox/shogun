/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_TYPE_TRAITS_H
#define SHOGUN_TYPE_TRAITS_H

namespace shogun
{
	template <typename T>
	struct get_container_underlying
	{
		using type = T;
	};

	template <template <typename> class Container, typename T>
	struct get_container_underlying<Container<T>>
	{
		using type = T;
	};

	template <typename T>
	using get_container_underlying_t = typename get_container_underlying<typename std::remove_reference<T>::type>::type;

	template <typename T, typename... Rest>
	struct sg_is_any_base_of : std::false_type
	{
	};

	template <typename T, typename First>
	struct sg_is_any_base_of<T, First> : std::is_base_of<T, get_container_underlying_t<First>>
	{
	};

	template <typename T, typename First, typename... Rest>
	struct sg_is_any_base_of<T, First, Rest...>
	    : std::integral_constant<
	          bool, std::is_base_of<T, get_container_underlying_t<First>>::value ||
	                    sg_is_any_base_of<T, Rest...>::value>
	{
	};

	template <typename T, bool, size_t size, typename... Rest>
	struct get_idx_from_pack_helper
	{
		constexpr static size_t idx = size;
	};

	template <typename T, size_t size, typename First>
	struct get_idx_from_pack_helper<T, false, size, First>
	{
		static_assert(std::is_base_of<T, get_container_underlying_t<First>>::value, "Could not find type in pack!\n");
		constexpr static size_t idx = size;
	};

	template <typename T, size_t size, typename First, typename... Rest>
	struct get_idx_from_pack_helper<T, true, size, First, Rest...>
	{
		constexpr static size_t idx = size - sizeof...(Rest) - 1;
	};

	template <typename T, size_t size, typename First, typename... Rest>
	struct get_idx_from_pack_helper<T, false, size, First, Rest...>
	{
		constexpr static size_t idx = get_idx_from_pack_helper<
				T, std::is_base_of<T, get_container_underlying_t<First>>::value, size, Rest...>::idx;
	};

	template <typename T, typename First, typename... Rest>
	struct get_idx_from_pack
	{
		constexpr static size_t idx = get_idx_from_pack_helper<
				T, std::is_base_of<T, get_container_underlying_t<First>>::value, sizeof...(Rest), Rest...>::idx;
	};

} // namespace shogun

#endif // SHOGUN_TYPE_TRAITS_H
