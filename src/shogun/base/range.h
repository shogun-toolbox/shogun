#ifndef __SG_RANGE_H__
#define __SG_RANGE_H__

#include <shogun/lib/config.h>
#include <iterator>

#ifdef HAVE_CXX11
namespace shogun
{

    /** @class Helper class to spawn range iterator.
     *
     * Useful for C++11-style for loops:
     *
     * @code
     *  for (auto i : Range(3, 10)) { ... }
     * @endcode
     */
    template <typename T>
    class Range
    {
        public:
            /** Creates range with specified bounds.
             * Assumes rbegin < rend.
             *
             * @param   rbegin   lower bound of range
             * @param   rend     upper bound of range (excluding)
             */
            Range(T rbegin, T rend) : m_begin(rbegin), m_end(rend)
            {
            }

            /** @class Iterator spawned by @ref Range. */
            class Iterator : public std::iterator<std::input_iterator_tag, T>
            {
                public:
                    Iterator(T value) : m_value(value)
                    {
                    }
                    Iterator(const Iterator& other) : m_value(other.m_value)
                    {
                    }
                    Iterator(Iterator&& other) : m_value(other.m_value)
                    {
                    }
                    Iterator& operator=(const Iterator&) = delete;
                    Iterator& operator++()
                    {
                        m_value++;
                        return *this;
                    }
                    Iterator& operator++(int)
                    {
                        Iterator tmp(*this);
                        tmp++;
                        return tmp; 
                    }
                    T operator*()
                    {
                        return m_value;
                    }
                    bool operator!=(const Iterator& other)
                    {
                        return this->m_value != other.m_value;
                    }
                    bool operator==(const Iterator& other)
                    {
                        return this->m_value == other.m_value;
                    }
                private:
                    T m_value;
            };
            /** Create iterator that corresponds to the start of range.
             *
             * Usually called through for-loop syntax.
             */
            Iterator begin() const
            {
                return Iterator(m_begin);
            }
            /** Create iterator that corresponds to the end of range.
             *
             * Usually called through for-loop syntax.
             */
            Iterator end() const
            {
                return Iterator(m_end);
            }
        private:
            /** begin of range */
            T m_begin;
            /** end of range */
            T m_end;
    };

    /** Creates @ref Range with specified upper bound.
     *
     * @code
     *  for (auto i : range(100)) { ... }
     * @endcode
     *
     * @param   rend     upper bound of range (excluding)
     */
    template <typename T>
    inline Range<T> range(T rend)
    {
        return Range<T>(0, rend);
    }

    /** Creates @ref Range with specified bounds.
     *
     * @code
     *  for (auto i : range(0, 100)) { ... }
     * @endcode
     *
     * @param   rbegin  lower bound of range
     * @param   rend    upper bound of range (excluding)
     */
    template <typename T>
    inline Range<T> range(T rbegin, T rend)
    {
        return Range<T>(rbegin, rend);
    }

}

#endif /* HAVE_CXX */
#endif /* __SG_RANGE_H__ */
