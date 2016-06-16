#ifndef __SG_SOME_H__
#define __SG_SOME_H__

#include <shogun/lib/config.h>

#ifdef HAVE_CXX11
#include <memory>

#include <shogun/base/SGObject.h>

namespace shogun
{

    /** @class Shogun synonym for the std::shared_ptr. Employs
     * exactly the same strategy for reference counting
     * as std::shared_ptr: any operation involving copying increases
     * the count and once deleted this wrapper decreases the counter.
     *
     */
    template <typename T>
        class Some
        {
            public:
                Some(const Some<T>& other);
                explicit Some(T* other);
                Some& operator=(T* other);
                ~Some();

                static Some<T> from_raw(T* raw);

                /** Casts the underlying object back to raw pointer
                 *
                 * Be careful to SG_REF obtained pointer if you start to own it.
                 *
                 * @return raw pointer (without SG_REF)
                 */
                operator T*();
                /** Call member function or access member of T
                 *
                 * @return raw pointer (without SG_REF)
                 */
                T* operator->();
            private:
                Some();
                void unref();
                void ref();
            private:
                T* raw;
        };

    template <typename T>
    Some<T>::Some()
    : raw(nullptr)
    {
    }
    template <typename T>
    Some<T>::Some(const Some<T>& other)
    : raw(other.raw)
    {
        ref();
    }
    template <typename T>
    Some<T>::Some(T* other)
    : raw(other)
    {
        ref();
    }
    template <typename T>
    Some<T>& Some<T>::operator=(T* other)
    {
        if (raw != other) {
            unref();
            raw = other;
            ref();
        }
        return *this;
    }
    template <typename T>
    Some<T>::~Some()
    {
        unref();
    }
    template <typename T>
    Some<T>::operator T*()
    {
        return raw;
    }
    template <typename T>
    T* Some<T>::operator->()
    {
        return raw;
    }
    template <typename T>
    void Some<T>::ref()
    {
        SG_REF(raw);
    }
    template <typename T>
    void Some<T>::unref()
    {
        SG_UNREF(raw);
    }
    template <typename T>
    Some<T> Some<T>::from_raw(T* raw)
    {
        Some<T> result(raw);
        return result;
    }

    /** Creates an instance of any class
     * that is wrapped with a shared pointer like
     * structure @ref Some
     *
     * @param args arguments to construct instance of T with (T should
     * have compatible constructor)
     *
     * @return a shared pointer that holds created instance of @ref T
     *
     */
    template <typename T, class... Args>
    Some<T> some(Args&&... args)
    {
        T* ptr = new T(args...);
        return Some<T>::from_raw(ptr);
    }

    template <class T>
    inline T wrap(const T& value)
    {
        return value;
    }

    template <class T>
    inline Some<T> wrap(T* ptr)
    {
        return Some<T>::from_raw(ptr);
    }

    template <class T>
    inline Some<T> wrap(const Some<T>& other)
    {
        return other;
    }

};

#endif /* HAVE_CXX11 */
#endif /* __SG_SOME_H__ */
