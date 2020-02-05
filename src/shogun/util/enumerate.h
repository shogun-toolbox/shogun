#include <shogun/util/zip_iterator.h>
#include <shogun/base/range.h>


namespace shogun {

	template <typename... Args>
	IGNORE_IN_CLASSLIST class enumerate
	{
	public:

		enumerate(Args&... args) : m_container_tuples(args...)
		{
		}

		template <typename... IteratorTypes>
		IGNORE_IN_CLASSLIST class Enumerate: public zip_iterator_detail::ZipIteratorBase<Enumerate<IteratorTypes...>, IteratorTypes...>
		{
		public:
			using iterator_category = std::forward_iterator_tag;


			Enumerate(std::tuple<IteratorTypes...>&& iterators)
			    : zip_iterator_detail::ZipIteratorBase<Enumerate<IteratorTypes...>, IteratorTypes...>(std::move(iterators)), m_index(0)
			{
			}

			Enumerate<IteratorTypes...>& operator++()
			{
				zip_iterator_detail::increment_iterators(
				    this->m_iterators_tuple,
				    std::index_sequence_for<IteratorTypes...>{});
				m_index++;
				return *this;
			}

			const Enumerate<IteratorTypes...> operator++(int)
			{
				Enumerate<IteratorTypes...> retval(this);
				++(this);
				return retval;
			}

			auto operator*()
			{
				return std::tuple_cat(std::make_tuple(m_index), zip_iterator_detail::dereference_iterators(
				    this->m_iterators_tuple,
				    std::index_sequence_for<IteratorTypes...>{}));
			}

		private:
			size_t m_index;
		};

#ifdef __clang__
		template <typename... IteratorTypes>
		Enumerate(std::tuple<IteratorTypes...>)
		    ->Enumerate<IteratorTypes...>;
#endif

		auto begin() const
		{
			return Enumerate(zip_iterator_detail::containers_begin(
			    m_container_tuples, std::index_sequence_for<Args...>{}));
		}

		auto end() const
		{
			return Enumerate(zip_iterator_detail::containers_end(
			    m_container_tuples, std::index_sequence_for<Args...>{}));
		}

	private:
		std::tuple<Args&...> m_container_tuples;
	};
}