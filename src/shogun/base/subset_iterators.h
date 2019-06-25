/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_SUBSET_ITERATORS_H
#define SHOGUN_SUBSET_ITERATORS_H

#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/Math.h>

namespace shogun
{
	template <typename T>
	class CDenseFeatures;
	class CDenseLabels;

	class CFeatures;
	class CLabels;

	namespace subset_iterator_detail
	{
		template <typename T>
		struct remove_cvptr
		{
			typedef std::remove_cv_t<std::remove_pointer_t<T>> type;
		};

		template <typename T>
		using remove_cvptr_t = typename remove_cvptr<T>::type;

		template <typename Container, typename T>
		struct return_type_subset_iterator
		{
		};

		template <typename T1, typename T2>
		struct return_type_subset_iterator<CDenseFeatures<T1>, T2>
		{
			using type = SGVector<T1>;
		};

		template <typename T>
		struct return_type_subset_iterator<CDenseLabels, T>
		{
			using type = float64_t;
		};
	} // namespace subset_iterator_detail

	template <class IterableSubsetContainer, typename ST>
	class SubsetIteratorBase
	{
	public:
		template <typename T, bool is_const = false>
		class subset_iterator
		{
		public:
			using iterator_category = std::forward_iterator_tag;
			using value_type = typename std::conditional_t<
			    is_const,
			    const typename subset_iterator_detail::
			        return_type_subset_iterator<
			            subset_iterator_detail::remove_cvptr_t<T>, T>::type,
			    typename subset_iterator_detail::return_type_subset_iterator<
			        subset_iterator_detail::remove_cvptr_t<T>, T>::type>;
			using difference_type = index_t;
			// not a reference but is used by stl algorithms to check type
			using reference = typename std::conditional_t<
			    is_const,
			    const typename subset_iterator_detail::
			        return_type_subset_iterator<
			            subset_iterator_detail::remove_cvptr_t<T>, T>::type,
			    typename subset_iterator_detail::return_type_subset_iterator<
			        subset_iterator_detail::remove_cvptr_t<T>, T>::type>;
			using pointer = typename std::conditional_t<
			    is_const,
			    const typename subset_iterator_detail::
			        return_type_subset_iterator<
			            subset_iterator_detail::remove_cvptr_t<T>, T>::type*,
			    typename subset_iterator_detail::return_type_subset_iterator<
			        subset_iterator_detail::remove_cvptr_t<T>, T>::type*>;

			using internal_pointer = typename std::conditional_t<
			    is_const, const subset_iterator_detail::remove_cvptr_t<T>*,
			    subset_iterator_detail::remove_cvptr_t<T>*>;

			subset_iterator(internal_pointer ptr, index_t idx = 0)
			{
				m_ptr = ptr;
				m_idx = idx;
			}

			subset_iterator(const subset_iterator<T, is_const>& other):  m_idx(other.m_idx), m_ptr(other.m_ptr)
			{
			}

			subset_iterator(subset_iterator<T, is_const>&& other) noexcept: m_idx(other.m_idx), m_ptr(other.m_ptr)
			{
			    other.m_idx = -1;
				other.m_ptr = nullptr;
			}

			subset_iterator<T, is_const>& operator++()
			{
				++m_idx;
				return *this;
			}

			subset_iterator<T, is_const> operator++(int)
			{
				subset_iterator<T, is_const> retval(*this);
				++(*this);
				return retval;
			}

			bool operator==(const subset_iterator<T, is_const>& other)
			{
				return m_ptr == other.m_ptr &&
				       m_idx == other.m_idx;
			}

			bool operator!=(const subset_iterator<T, is_const>& other)
			{
				return !(*this == other);
			}

			reference operator*()
			{
				if constexpr (std::is_base_of<
				                  CFeatures, subset_iterator_detail::
				                                 remove_cvptr_t<T>>::value)
					return m_ptr->get_feature_vector(m_idx);
				if constexpr (std::is_base_of<
				                  CLabels, subset_iterator_detail::
				                               remove_cvptr_t<T>>::value)
					return m_ptr->get_label(m_idx);
			}

		private:
			index_t m_idx;
			internal_pointer m_ptr;
		};

		/**
		 * Returns a const iterator to the first vector of features/labels.
		 */
		auto begin() const
		{
			const auto* this_casted =
			    static_cast<const IterableSubsetContainer*>(this);
			return subset_iterator<decltype(this_casted), true>(this_casted);
		}

		/**
		 * Returns a const iterator to the "vector" following the end of
		 * features/labels.
		 */
		auto end() const
		{
			const auto* this_casted =
			    static_cast<const IterableSubsetContainer*>(this);
			if (auto stack = this_casted->get_subset_stack()->get_last_subset();
			    stack == nullptr)
			{
				if constexpr (std::is_base_of<
				                  CFeatures,
				                  subset_iterator_detail::remove_cvptr_t<
				                      IterableSubsetContainer>>::value)
					return subset_iterator<decltype(this_casted), true>(
					    this_casted, this_casted->get_num_vectors());
				if constexpr (std::is_base_of<
				                  CLabels,
				                  subset_iterator_detail::remove_cvptr_t<
				                      IterableSubsetContainer>>::value)
					return subset_iterator<decltype(this_casted), true>(
					    this_casted, this_casted->get_num_labels());
			}
			else
				return subset_iterator<decltype(this_casted), true>(
				    this_casted, stack->get_subset_idx().size());
		}
	};
} // namespace shogun
#endif // SHOGUN_SUBSET_ITERATORS_H
