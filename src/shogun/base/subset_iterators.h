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
	struct remove_cvptr
	{
		typedef std::remove_cv_t<std::remove_pointer_t<T>> type;
	};

	template <typename T>
	using remove_cvptr_t = typename remove_cvptr<T>::type;

	template <typename T>
	class CDenseFeatures;
    class CDenseLabels;

	class CFeatures;
	class CLabels;

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
			    const typename return_type_subset_iterator<
			        remove_cvptr_t<T>, T>::type,
			    typename return_type_subset_iterator<remove_cvptr_t<T>, T>::type>;
			using difference_type = index_t;
			// not a reference but is used by stl algorithms to check type
			using reference = typename std::conditional_t<
			    is_const,
			    const typename return_type_subset_iterator<
			        remove_cvptr_t<T>, T>::type,
			    typename return_type_subset_iterator<remove_cvptr_t<T>, T>::type>;
			using pointer = typename std::conditional_t<
			    is_const,
			    const typename return_type_subset_iterator<
			        remove_cvptr_t<T>, T>::type*,
			    typename return_type_subset_iterator<remove_cvptr_t<T>, T>::type*>;

			using internal_pointer = typename std::conditional_t<
			    is_const, const remove_cvptr_t<T>*, remove_cvptr_t<T>*>;

			subset_iterator(internal_pointer ptr, index_t idx = 0)
			{
				if (ptr->get_subset_stack()->get_last_subset() != nullptr)
					m_subset = ptr->get_subset_stack()
					               ->get_last_subset()
					               ->get_subset_idx();
				m_idx = idx;
				// need to keep this variable so that the vector destructor
				// isn't called
				m_argsorted_subset = CMath::argsort(m_subset);
				m_argsorted_subset_iter = m_argsorted_subset.begin() + idx;
				m_ptr = ptr;
			}

			subset_iterator<T, is_const>& operator++()
			{
				m_argsorted_subset_iter++;
				m_idx++;
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
				return m_ptr == other.m_ptr && m_idx == other.m_idx;
			}

			bool operator!=(const subset_iterator<T, is_const>& other)
			{
				return !(*this == other);
			}

			reference operator*()
			{
				index_t idx;
				if (m_ptr->get_subset_stack()->get_last_subset() != nullptr)
					idx = *m_argsorted_subset_iter;
				else
					idx = m_idx;
				if constexpr (std::is_base_of<
				                  CFeatures, remove_cvptr_t<T>>::value)
					return m_ptr->get_feature_vector(idx);
				if constexpr (std::is_base_of<
				                  CLabels, remove_cvptr_t<T>>::value)
					return m_ptr->get_label(idx);
			}

		private:
			index_t m_idx;
			internal_pointer m_ptr;
			SGVector<index_t> m_subset;
			SGVector<index_t> m_argsorted_subset;
			SGVector<index_t>::iterator m_argsorted_subset_iter;
		};

		/**
		 * Returns an iterator to the first vector of features/labels.
		 */
		auto begin()
		{
			auto* this_casted = static_cast<IterableSubsetContainer*>(this);
			return subset_iterator<decltype(this_casted)>(this_casted);
		}

		/**
		 * Returns an iterator to the element following the end of
		 * features/labels.
		 */
		auto end()
		{
			auto* this_casted = static_cast<IterableSubsetContainer*>(this);
			if (auto stack = this_casted->get_subset_stack()->get_last_subset();
			    stack == nullptr)
			{
				if constexpr (std::is_base_of<
				                  CFeatures,
				                  remove_cvptr_t<IterableSubsetContainer>>::
				                  value)
					return subset_iterator<decltype(this_casted)>(
					    this_casted, this_casted->get_num_vectors());
				if constexpr (std::is_base_of<
				                  CLabels, remove_cvptr_t<
				                               IterableSubsetContainer>>::value)
					return subset_iterator<decltype(this_casted)>(
					    this_casted, this_casted->get_num_labels());
			}
			else
				return subset_iterator<decltype(this_casted)>(
				    this_casted, stack->get_subset_idx().size());
		}

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
		 * Returns a const iterator to the element following the end of
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
				                  remove_cvptr_t<IterableSubsetContainer>>::
				                  value)
					return subset_iterator<decltype(this_casted), true>(
					    this_casted, this_casted->get_num_vectors());
				if constexpr (std::is_base_of<
				                  CLabels, remove_cvptr_t<
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
