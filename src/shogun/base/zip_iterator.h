/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_ZIP_ITERATOR_H
#define SHOGUN_ZIP_ITERATOR_H

#include <tuple>

namespace shogun
{
	namespace zip_iterator_detail
	{
		template <typename... Args1, typename... Args2, size_t... Idx>
		bool iterators_equal(
		    std::tuple<Args1...>& container1,
		    const std::tuple<Args2...>& container2, std::index_sequence<Idx...>)
		{
			return (
			    (std::get<Idx>(container1) == std::get<Idx>(container2)) &&
			    ...);
		}

		template <typename... Args, size_t... Idx>
		void increment_iterators(
		    std::tuple<Args...>& container, std::index_sequence<Idx...>)
		{
			((++std::get<Idx>(container)), ...);
		}

		template <typename... Args, size_t... Idx>
		auto dereference_iterators(
		    std::tuple<Args...>& container, std::index_sequence<Idx...>)
		{
			return std::make_tuple(*(std::get<Idx>(container))...);
		}

		template <typename T>
		auto get_begin(T& container) -> decltype(container.begin())
		{
			return container.begin();
		}

		template <typename T>
		auto get_begin(T& container) -> decltype(container->begin())
		{
			return container->begin();
		}

		template <typename T>
		auto get_end(T& container) -> decltype(container.end())
		{
			return container.end();
		}

		template <typename T>
		auto get_end(T& container) -> decltype(container->end())
		{
			return container->end();
		}

		template <typename... Args, size_t... Idx>
		auto containers_begin(
		    std::tuple<Args...>& container, std::index_sequence<Idx...>)
		{
			return std::make_tuple((get_begin(std::get<Idx>(container)))...);
		}

		template <typename... Args, size_t... Idx>
		auto containers_end(
		    std::tuple<Args...>& container, std::index_sequence<Idx...>)
		{
			return std::make_tuple((get_end(std::get<Idx>(container)))...);
		}
	} // namespace zip_iterator_detail
	template <typename... Args>
	class zip_iterator
	{
	public:
		zip_iterator(const Args&... args) : m_iterator_tuples(args...)
		{
		}

		template <typename... ZipTypeArgs>
		class ZipIterator
		{
		public:
			using iterator_category = std::forward_iterator_tag;

			ZipIterator(std::tuple<ZipTypeArgs...>&& values)
			    : m_value_tuple(std::move(values))
			{
			}

			ZipIterator<ZipTypeArgs...>& operator++()
			{
				zip_iterator_detail::increment_iterators(
				    m_value_tuple, std::index_sequence_for<ZipTypeArgs...>{});
				return *this;
			}

			const ZipIterator<ZipTypeArgs...> operator++(int)
			{
				ZipIterator<ZipTypeArgs...> retval(this);
				++(this);
				return retval;
			}

			auto operator*()
			{
				return zip_iterator_detail::dereference_iterators(
				    m_value_tuple, std::index_sequence_for<ZipTypeArgs...>{});
			}

			bool operator==(const ZipIterator& other)
			{
				return zip_iterator_detail::iterators_equal(
				    m_value_tuple, other.m_value_tuple,
				    std::index_sequence_for<ZipTypeArgs...>{});
			}

			bool operator!=(const ZipIterator& other)
			{
				return !(*this == other);
			}

		private:
			std::tuple<ZipTypeArgs...> m_value_tuple;
		};

		// conveniently gcc doesn't need a deduction guide
		// however there is a bug where "subobjects" cannot have a deduction
		// guide in gcc
#ifdef __clang__
		template <typename... ZipTypeArgs>
		ZipIterator(std::tuple<ZipTypeArgs...>)->ZipIterator<ZipTypeArgs...>;
#endif

		auto begin()
		{
			return ZipIterator(zip_iterator_detail::containers_begin(
			    m_iterator_tuples, std::index_sequence_for<Args...>{}));
		}

		auto end()
		{
			return ZipIterator(zip_iterator_detail::containers_end(
			    m_iterator_tuples, std::index_sequence_for<Args...>{}));
		}

	private:
		std::tuple<const Args&...> m_iterator_tuples;
	};
} // namespace shogun

#endif // SHOGUN_ZIP_ITERATOR_H
