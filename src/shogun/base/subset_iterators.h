/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_SUBSET_ITERATORS_H
#define SHOGUN_SUBSET_ITERATORS_H

#include <shogun/features/Features.h>

namespace shogun {
    template<typename T, std::enable_if<
            (std::is_base_of<T, CFeatures>::value) || (std::is_base_of<T, CLabels>::value)> * = nullptr>
    class SubsetIterator {
    public:
        SubsetIterator(T *pointer) {
            if (pointer->get_subset_stack()->get_last_subset() != nullptr)
                m_subset = pointer->get_subset_stack()->get_last_subset()->get_subset_idx();
            m_idx = 0;
            m_argsorted_subset = CMath::argsort(m_subset);
            m_argsorted_subset_iter = m_argsorted_subset.begin();
            m_ptr = pointer;
            SG_REF(m_ptr)
        }

        ~SubsetIterator() {
            SG_UNREF(m_ptr)
        }

        SubsetIterator operator++(int) {
            m_argsorted_subset_iter++;
            m_idx++;
            return *this;
        }

        SubsetIterator& operator++() {
            m_argsorted_subset_iter++;
            m_idx++;
            return *this;
        }

        const auto operator*() {
            index_t idx;
            if (m_ptr->get_subset_stack()->get_last_subset() != nullptr)
                idx = m_subset[*m_argsorted_subset_iter];
            else
                idx = m_idx;
            if constexpr (std::is_base_of<CFeatures, T>::value)
                return m_ptr->get_feature_vector(idx);
            if constexpr (std::is_base_of<CLabels, T>::value)
                return m_ptr->get_label(idx);
        }

    private:
        index_t m_idx;
        T *m_ptr;
        SGVector<index_t> m_subset;
        SGVector<index_t> m_argsorted_subset;
        SGVector<index_t>::iterator m_argsorted_subset_iter;
    };
}
#endif //SHOGUN_SUBSET_ITERATORS_H
