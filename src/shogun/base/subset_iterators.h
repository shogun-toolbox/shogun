/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_SUBSET_ITERATORS_H
#define SHOGUN_SUBSET_ITERATORS_H

#include <shogun/features/Features.h>
#include <shogun/labels/Labels.h>

namespace shogun
{
	template <typename T>
	class CDenseFeatures;

	//    template <typename T>
	//    class CDenseLabels;

	template <typename T>
	struct return_type_subset_iterator
	{
	};

	template <typename T>
	struct return_type_subset_iterator<CDenseFeatures<T>>
	{
		using type = SGVector<T>;
	};

	//    template <typename T>
	//    struct return_type_subset_iterator<CDenseLabels<T>>
	//    {
	//        using type = T;
	//    };

	template <
	    typename T, std::enable_if<
	                    (std::is_base_of<T, CFeatures>::value) ||
	                    (std::is_base_of<T, CLabels>::value)>* = nullptr>
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
} // namespace shogun
#endif // SHOGUN_SUBSET_ITERATORS_H
