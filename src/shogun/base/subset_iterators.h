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

	class CFeatures;
	class CLabels;

	template <typename T>
	struct return_type_subset_iterator
	{
	};

	template <typename T>
	struct return_type_subset_iterator<CDenseFeatures<T>>
	{
		using type = SGVector<T>;
	};

	template <
	    typename T, std::enable_if<
	                    (std::is_base_of_v<T, CFeatures>) ||
	                    (std::is_base_of_v<T, CLabels>)>* = nullptr>
	class SubsetIterator
	{
	public:
		using iterator_category = std::forward_iterator_tag;
		using value_type = typename return_type_subset_iterator<T>::type;
		using difference_type = index_t;
		// not a reference but is used by stl algorithms to check type
		using reference = typename return_type_subset_iterator<T>::type;
		using pointer = typename return_type_subset_iterator<T>::type*;

		SubsetIterator(T* pointer, index_t idx = 0)
		{
			if (pointer->get_subset_stack()->get_last_subset() != nullptr)
				m_subset = pointer->get_subset_stack()
				               ->get_last_subset()
				               ->get_subset_idx();
			m_idx = idx;
			// need to keep this variable so that the vector destructor isn't
			// called
			m_argsorted_subset = CMath::argsort(m_subset);
			m_argsorted_subset_iter = m_argsorted_subset.begin() + idx;
			m_ptr = pointer;
		}

		SubsetIterator& operator++()
		{
			m_argsorted_subset_iter++;
			m_idx++;
			return *this;
		}

		SubsetIterator operator++(int)
		{
			SubsetIterator retval(*this);
			++(*this);
			return retval;
		}

		bool operator==(const SubsetIterator<T>& other)
		{
			return m_ptr == other.m_ptr && m_idx == other.m_idx;
		}

		bool operator!=(const SubsetIterator<T>& other)
		{
			return !(*this == other);
		}

		const reference operator*()
		{
			index_t idx;
			if (m_ptr->get_subset_stack()->get_last_subset() != nullptr)
				idx = *m_argsorted_subset_iter;
			else
				idx = m_idx;
			if constexpr (std::is_base_of<CFeatures, T>::value)
				return m_ptr->get_feature_vector(idx);
			if constexpr (std::is_base_of<CLabels, T>::value)
				return m_ptr->get_label(idx);
		}

	private:
		index_t m_idx;
		T* m_ptr;
		SGVector<index_t> m_subset;
		SGVector<index_t> m_argsorted_subset;
		SGVector<index_t>::iterator m_argsorted_subset_iter;
	};

	template <template <typename> class IterableSubsetContainer, typename T>
	class SubsetIteratorBase
	{
	public:
		using iterator = SubsetIterator<IterableSubsetContainer<T>>;

		/**
		 * Returns an iterator to the first vector of features.
		 */
		iterator begin() noexcept
		{
			return SubsetIterator(
			    static_cast<IterableSubsetContainer<T>*>(this));
		}

		/**
		 * Returns an iterator to the element following the end of features.
		 */
		iterator end() noexcept
		{
			auto this_casted = static_cast<IterableSubsetContainer<T>*>(this);
			if (auto stack = this_casted->get_subset_stack()->get_last_subset();
			    stack == nullptr)
				return SubsetIterator(
				    this_casted, this_casted->get_num_vectors());
			else
				return SubsetIterator(
				    this_casted, stack->get_subset_idx().size());
		}
	};

} // namespace shogun
#endif // SHOGUN_SUBSET_ITERATORS_H
