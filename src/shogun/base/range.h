#ifndef __SG_RANGE_H__
#define __SG_RANGE_H__

#include <iterator>
#include <iostream>

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
            /** Creates range.
             * Assumes start < end.
             *
             * @param   start   start of range
             * @param   end     end of range
             */
            Range(T start, T end) : m_start(start), m_end(end)
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
                return Iterator(m_start);
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
            /** start of range */
            T m_start;
            /** end of range */
            T m_end;
    };

    /** Creates @ref Range with specified upper boundary.
     *
     * @code
     *  for (auto i : range(100)) { ... }
     * @endcode
     *
     * @param   end     end of range
     */
    template <typename T>
    inline Range<T> range(T end)
    {
        return Range<T>(0, end);
    }

    /** Creates @ref Range with specified boundaries.
     *
     * @code
     *  for (auto i : range(0, 100)) { ... }
     * @endcode
     *
     * @param   start   start of range
     * @param   end     end of range
     */
    template <typename T>
    inline Range<T> range(T start, T end)
    {
        return Range<T>(start, end);
    }

}

#endif /* __SG_RANGE_H__ */
