/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_ZIP_ITERATOR_H
#define SHOGUN_ZIP_ITERATOR_H

#include <shogun/base/macros.h>
#include <tuple>

namespace shogun
{
	namespace zip_iterator_detail
	{
		template <typename... Args1, typename... Args2, size_t... Idx>
		SG_FORCED_INLINE bool iterators_equal(
		    const std::tuple<Args1...>& iterators1,
		    const std::tuple<Args2...>& iterators2, std::index_sequence<Idx...>)
		{
			return (
			    (std::get<Idx>(iterators1) == std::get<Idx>(iterators2)) &&
			    ...);
		}

		template <typename... Args, size_t... Idx>
		SG_FORCED_INLINE void increment_iterators(
		    std::tuple<Args...>& iterators, std::index_sequence<Idx...>)
		{
			((++std::get<Idx>(iterators)), ...);
		}

		template <typename... Args, size_t... Idx>
		SG_FORCED_INLINE auto dereference_iterators(
		    std::tuple<Args...>& iterators, std::index_sequence<Idx...>)
		{
			return std::make_tuple(*(std::get<Idx>(iterators))...);
		}

		template <typename T>
		SG_FORCED_INLINE auto get_begin(const T& container)
		    -> decltype(container.begin())
		{
			return container.begin();
		}

		template <typename T>
		SG_FORCED_INLINE auto get_begin(const T& container)
		    -> decltype(container->begin())
		{
			return container->begin();
		}

		template <typename T>
		SG_FORCED_INLINE auto get_end(const T& container) -> decltype(container.end())
		{
			return container.end();
		}

		template <typename T>
		SG_FORCED_INLINE auto get_end(const T& container)
		    -> decltype(container->end())
		{
			return container->end();
		}

		template <typename... Args, size_t... Idx>
		SG_FORCED_INLINE auto containers_begin(
		    const std::tuple<Args...>& container, std::index_sequence<Idx...>)
		{
			return std::make_tuple((get_begin(std::get<Idx>(container)))...);
		}

		template <typename... Args, size_t... Idx>
		SG_FORCED_INLINE auto containers_end(
		    const std::tuple<Args...>& container, std::index_sequence<Idx...>)
		{
			return std::make_tuple((get_end(std::get<Idx>(container)))...);
		}
	} // namespace zip_iterator_detail
	template <typename... Args>
	class zip_iterator
	{
	public:
		zip_iterator(Args&... args) : m_container_tuples(args...)
		{
		}

		template <typename... IteratorTypes>
		class ZipIterator
		{
		public:
			using iterator_category = std::forward_iterator_tag;


			ZipIterator(std::tuple<IteratorTypes...>&& iterators)
			    : m_iterators_tuple(std::move(iterators))
			{
			}

			ZipIterator<IteratorTypes...>& operator++()
			{
				zip_iterator_detail::increment_iterators(
				    m_iterators_tuple,
				    std::index_sequence_for<IteratorTypes...>{});
				return *this;
			}

			const ZipIterator<IteratorTypes...> operator++(int)
			{
				ZipIterator<IteratorTypes...> retval(this);
				++(this);
				return retval;
			}

			auto operator*()
			{
				return zip_iterator_detail::dereference_iterators(
				    m_iterators_tuple,
				    std::index_sequence_for<IteratorTypes...>{});
			}

			bool operator==(const ZipIterator& other) const
			{
				return zip_iterator_detail::iterators_equal(
				    m_iterators_tuple, other.m_iterators_tuple,
				    std::index_sequence_for<IteratorTypes...>{});
			}

			bool operator!=(const ZipIterator& other) const
			{
				return !(*this == other);
			}

		private:
			std::tuple<IteratorTypes...> m_iterators_tuple;
		};

		// conveniently gcc doesn't need a deduction guide
		// however there is a bug where "subobjects" cannot have a deduction
		// guide in gcc
#ifdef __clang__
		template <typename... IteratorTypes>
		ZipIterator(std::tuple<IteratorTypes...>)
		    ->ZipIterator<IteratorTypes...>;
#endif

		auto begin() const
		{
			return ZipIterator(zip_iterator_detail::containers_begin(
			    m_container_tuples, std::index_sequence_for<Args...>{}));
		}

		auto end() const
		{
			return ZipIterator(zip_iterator_detail::containers_end(
			    m_container_tuples, std::index_sequence_for<Args...>{}));
		}

	private:
		std::tuple<Args&...> m_container_tuples;
	};
} // namespace shogun

#endif // SHOGUN_ZIP_ITERATOR_H
