/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_ZIP_ITERATOR_H
#define SHOGUN_ZIP_ITERATOR_H

#include <memory>
#include <tuple>
#include <utility>

namespace shogun
{
	template <typename T>
	struct is_shared_ptr : public std::false_type
	{
	};
	template <typename T>
	struct is_shared_ptr<std::shared_ptr<T>> : public std::true_type
	{
	};

	template <typename... Args1, typename... Args2, size_t... Idx>
	bool compare_containers(
	    std::tuple<Args1...> container1, std::tuple<Args2...> container2,
	    std::index_sequence<Idx...>)
	{
		return (
		    (std::get<Idx>(container1) == std::get<Idx>(container2)) && ...);
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

	template <typename... Args>
	class zip_iterator
	{
	public:
		zip_iterator(Args... args)
		{
			m_iterator_tuples = std::forward_as_tuple(args...);
		}

		template <typename... ZipTypeArgs>
		class ZipIterator
		{
		public:
			ZipIterator(std::tuple<ZipTypeArgs...>&& values)
			    : m_value_tuple(std::move(values))
			{
			}

			ZipIterator<ZipTypeArgs...> operator++()
			{
				return std::apply(
				    [](auto&... container) {
					    return std::make_tuple(++container...);
				    },
				    m_value_tuple);
			}

			auto operator*()
			{
				return std::apply(
				    [](auto&... container) {
					    return std::make_tuple(*container...);
				    },
				    m_value_tuple);
			}

			bool operator==(const ZipIterator& other)
			{
				return compare_containers(
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
			return ZipIterator(std::apply(
			    [](auto... container) {
				    return std::make_tuple(get_begin(container)...);
			    },
			    m_iterator_tuples));
		}

		auto begin() const
		{
			return ZipIterator(std::apply(
			    [](const auto&... container) {
				    return std::make_tuple(get_begin(container)...);
			    },
			    m_iterator_tuples));
		}

		auto end()
		{
			return ZipIterator(std::apply(
			    [](auto... container) {
				    return std::make_tuple(get_end(container)...);
			    },
			    m_iterator_tuples));
		}

		auto end() const
		{
			return ZipIterator(std::apply(
			    [](const auto&... container) {
				    return std::make_tuple(get_end(container)...);
			    },
			    m_iterator_tuples));
		}

	private:
		std::tuple<Args...> m_iterator_tuples;
	};
} // namespace shogun

#endif // SHOGUN_ZIP_ITERATOR_H
