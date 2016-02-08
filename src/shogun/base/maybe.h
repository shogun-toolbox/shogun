#ifndef __SG_MAYBE_H__
#define __SG_MAYBE_H__

#include <shogun/lib/ShogunException.h>

namespace shogun
{

    namespace detail
    {
        static const char* NO_REASON_FOR_ABSENCE = "AVAILABLE";
    }

    /** Represents non-typed absent value.
     *
     * Can be casted to any @see Maybe<T> resulting
     * in Maybe with absent value.
     *
     * Contains reason for absence as a regular C string.
     */
    class Nothing
    {
    public:
        /** Createst an instance of nothing aka absent value.
         *
         * @param reason the reason something is absent
         */
        Nothing(const char* reason = "no specific reason") :
            m_absence_reason(reason)
        {
        }
        /** Copies nothing from nothing, inheriting absence reason.
         *
         * @param other other instance of nothing
         */
        Nothing(const Nothing& other) :
            m_absence_reason(other.m_absence_reason)
        {
        }
        /** Represents a reason why something is absent.
         * Memory is not managed.
         */
        const char* m_absence_reason;
    };

    /** @class Holder that represents an object that can be
     * either present or absent. Quite simllar to std::optional
     * introduced in C++14, but provides a way to pass the reason
     * of absence (e.g. "incorrect parameter").
     *
     * Essentially, instances are created via @see Just or @see Nothing
     *
     * @code
     *      Maybe<int> absent_value = Nothing();
     *      Maybe<int> present_value = Just(3);
     * @endcode
     *
     * To check whether value is present, regular implicit cast to bool is used:
     *
     * @code
     *      if (maybe_value)
     *      {
     *          // value exists!
     *      }
     *      else
     *      {
     *          // value doesn't exist!
     *      }
     * @endcode
     *
     * To obtain value, @see value is used:
     *
     * @code
     *      // may throw an exception
     *      int value = unreliable.value();
     * @endcode
     *
     * To provide default values, @see value_or is used:
     *
     * @code
     *      int value = unreliable.value_or(9);
     * @endcode
     *
     */
    template <typename T>
    class Maybe
    {
    public:
        /** Creates an instance from nothing resulting in absent value.
         *
         * @param   nothing     instance of nothing
         *
         */
        Maybe(const Nothing& nothing) :
            m_value(),
            m_absence_reason(nothing.m_absence_reason)
        {
        }
        /** Copy constructor.
         *
         * @param   other       other instance of Maybe for the same type
         */
        Maybe(const Maybe<T>& other) :
            m_value(other.m_value),
            m_absence_reason(other.m_absence_reason)
        {
        }

        /** Evaluates to true when object is present, false otherwise.
         *
         * Equivalent to @see is_present()
         */
        inline operator bool() const
        {
            return is_present();
        }

        /** Returns value if it is present, fails otherwise.
         *
         * @throw   ShogunException     if retrieved value is absent
         */
        inline T& value()
        {
            return *get();
        }

        /** Returns value if it is present, or the provided default value.
         *
         * Doesn't throw any exception.
         */
        inline T& value_or(T& v)
        {
            if (is_present())
                return *get();
            else
                return v;
        }

        /** Returns true if value is absent, false otherwise.
         *
         * Doesn't throw any exception.
         */
        inline bool is_absent() const
        {
            return m_absence_reason != detail::NO_REASON_FOR_ABSENCE;
        }

        /** Returns true if value is present, false otherwise.
         *
         * Doesn't throw any exception.
         */
        inline bool is_present() const
        {
            return m_absence_reason == detail::NO_REASON_FOR_ABSENCE;
        }

    private:
        inline T* get()
        {
            if (is_present())
                return &m_value;
            else
                throw ShogunException("Tried to access inexistent object");
        }
        Maybe();
        Maybe(const char* reason, bool) :
            m_value(),
            m_absence_reason(reason)
        {
        }
        Maybe(const T& value) :
            m_value(value),
            m_absence_reason(detail::NO_REASON_FOR_ABSENCE)
        {
        }

    public:
        template <typename Q>
        friend Maybe<Q> Just(const Q& value);

    private:
        T m_value;
        const char* m_absence_reason;
    };

    /** Wraps provided value into Maybe
     * with present value.
     *
     * Doesn't throw any exception.
     *
     * @param   value   the value to wrap
     */
    template <typename T>
    static Maybe<T> Just(const T& value)
    {
        return Maybe<T>(value);
    }

}
#endif
