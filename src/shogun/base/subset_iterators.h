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

		class SubsetIteratorHelperAbstract
		{
		public:
			SubsetIteratorHelperAbstract(index_t idx) : m_idx(idx)
			{
			}
			virtual void increment() = 0;
			virtual index_t get() = 0;
			SG_FORCED_INLINE index_t get_idx() const
			{
				return m_idx;
			}

		protected:
			index_t m_idx;
		};

		class SubsetIteratorHelperSubset : public SubsetIteratorHelperAbstract
		{
		public:
			template <typename T>
			SubsetIteratorHelperSubset(T* ptr, int idx)
			    : SubsetIteratorHelperAbstract(idx)
			{
				// need to keep this variable so that the vector destructor
				// isn't called
				m_argsorted_subset = CMath::argsort(ptr->get_subset_stack()
				                                        ->get_last_subset()
				                                        ->get_subset_idx());
				m_argsorted_subset_iter = m_argsorted_subset.begin() + idx;
			}

			void increment() final
			{
				m_argsorted_subset_iter++;
				m_idx++;
			}

			index_t get() final
			{
				return *m_argsorted_subset_iter;
			}

		private:
			SGVector<index_t> m_argsorted_subset;
			SGVector<index_t>::iterator m_argsorted_subset_iter;
		};

		class SubsetIteratorHelperLinear : public SubsetIteratorHelperAbstract
		{
		public:
			SubsetIteratorHelperLinear(index_t idx)
			    : SubsetIteratorHelperAbstract(idx)
			{
			}

			void increment() final
			{
				this->m_idx++;
			}

			index_t get() final
			{
				return m_idx;
			}
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
				if (ptr->get_subset_stack()->get_last_subset() != nullptr)
				{
					m_idx_holder = std::make_shared<
					    subset_iterator_detail::SubsetIteratorHelperSubset>(
					    ptr, idx);
				}
				else
				{
					m_idx_holder = std::make_shared<
					    subset_iterator_detail::SubsetIteratorHelperLinear>(
					    idx);
				}
				m_ptr = ptr;
			}

			subset_iterator(const subset_iterator<T, is_const>& other)
			{
				m_idx_holder = other.m_idx_holder;
				m_ptr = other.m_ptr;
			}

			subset_iterator(subset_iterator<T, is_const>&& other) noexcept
			{
				m_idx_holder = std::move(other.m_idx_holder);
				m_ptr = other.m_ptr;
			}

			subset_iterator<T, is_const>& operator++()
			{
				m_idx_holder->increment();
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
				       m_idx_holder->get_idx() == other.m_idx_holder->get_idx();
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
					return m_ptr->get_feature_vector(m_idx_holder->get());
				if constexpr (std::is_base_of<
				                  CLabels, subset_iterator_detail::
				                               remove_cvptr_t<T>>::value)
					return m_ptr->get_label(m_idx_holder->get());
			}

		private:
			std::shared_ptr<
			    subset_iterator_detail::SubsetIteratorHelperAbstract>
			    m_idx_holder;
			internal_pointer m_ptr;
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
				                  subset_iterator_detail::remove_cvptr_t<
				                      IterableSubsetContainer>>::value)
					return subset_iterator<decltype(this_casted)>(
					    this_casted, this_casted->get_num_vectors());
				if constexpr (std::is_base_of<
				                  CLabels,
				                  subset_iterator_detail::remove_cvptr_t<
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
