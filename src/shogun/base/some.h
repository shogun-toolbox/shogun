#ifndef __SG_SOME_H__
#define __SG_SOME_H__

#include <memory>
#include <shogun/base/macros.h>

namespace shogun
{

	/** @class Shogun synonym for the std::shared_ptr. Employs
	 * exactly the same strategy for reference counting
	 * as std::shared_ptr: any operation involving copying increases
	 * the count and once deleted this wrapper decreases the counter.
	 *
	 */
	template <class T>
	class Some
	{
	public:
		Some(const Some<T>& other);
		template <class R>
		Some(const Some<R>& other);

		Some(Some<T>&& other);
		template <class R>
		Some(Some<R>&& other);

		Some<T>& operator=(const Some<T>& other);
		template <class R>
		Some<T>& operator=(const Some<R>& other);

		Some<T>& operator=(Some<T>&& other);
		template <class R>
		Some<T>& operator=(Some<R>&& other);

		~Some();

		static Some<T> from_raw(T* raw);

		/** Casts the underlying object back to raw pointer
		 *
		 * Be careful to SG_REF obtained pointer if you start to own it.
		 *
		 * @return raw pointer (without SG_REF)
		 */
		operator T*() const;

		/** Call member function or access member of T
		 *
		 * @return raw pointer (without SG_REF)
		 */
		T* operator->() const;

		/** Equality operator
		 * @param other other element to compare with
		 * @return true iff other's raw pointer equals own raw pointer
		 */
		bool operator==(const Some<T>& other) const;

		/** Inequality operator
		 * @param other other element to compare with
		 * @return false iff other's raw pointer equals own raw pointer
		 */
		bool operator!=(const Some<T>& other) const;

		/**
		 * Get the raw pointer
		 *
		 * @return raw pointer (without SG_REF)
		 */
		T* get() const;

		void reset(T* value = nullptr);

	private:
		Some(T* other);
		Some();
		void unref();
		void ref();

	private:
		T* raw;
	};

	template <class T>
	Some<T>::Some() : raw(nullptr)
	{
	}
	template <class T>
	Some<T>::Some(T* other) : raw(other)
	{
		ref();
	}
	template <class T>
	template <class R>
	Some<T>::Some(const Some<R>& other) : raw(nullptr)
	{
		reset(dynamic_cast<T*>(other.get()));
		ref();
	}
	template <class T>
	Some<T>::Some(const Some<T>& other) : raw(other.get())
	{
		ref();
	}
	template <class T>
	template <class R>
	Some<T>::Some(Some<R>&& other) : raw(nullptr)
	{
		reset(dynamic_cast<T*>(other.get()));
		other.raw = nullptr;
	}
	template <class T>
	Some<T>::Some(Some<T>&& other) : raw(other.get())
	{
		other.raw = nullptr;
	}
	template <class T>
	template <class R>
	Some<T>& Some<T>::operator=(const Some<R>& other)
	{
		if (get() != other.get())
		{
			reset(other.get());
			ref();
		}
		return *this;
	}
	template <class T>
	Some<T>& Some<T>::operator=(const Some<T>& other)
	{
		if (get() != other.get())
		{
			reset(other.get());
			ref();
		}
		return *this;
	}
	template <class T>
	template <class R>
	Some<T>& Some<T>::operator=(Some<R>&& other)
	{
		if (get() != other.get())
		{
			reset(other.get());
			other.raw = nullptr;
		}
		return *this;
	}
	template <class T>
	Some<T>& Some<T>::operator=(Some<T>&& other)
	{
		if (get() != other.get())
		{
			reset(other.get());
			other.raw = nullptr;
		}
		return *this;
	}

	template <class T>
	Some<T>::~Some()
	{
		reset();
	}
	template <typename T>
	Some<T>::operator T*() const
	{
		return get();
	}
	template <class T>
	T* Some<T>::operator->() const
	{
		return get();
	}
	template <class T>
	bool Some<T>::operator==(const Some<T>& other) const
	{
		return raw == other.raw;
	}
	template <class T>
	bool Some<T>::operator!=(const Some<T>& other) const
	{
		return !((*this) == other);
	}
	template <class T>
	T* Some<T>::get() const
	{
		return raw;
	}
	template <class T>
	void Some<T>::reset(T* ptr)
	{
		unref();
		raw = ptr;
	}
	template <class T>
	void Some<T>::ref()
	{
		if (raw)
		{
			(raw)->ref();
		}
	}
	template <class T>
	void Some<T>::unref()
	{
		if (raw)
		{
			if ((raw)->unref() == 0)
				(raw) = NULL;
		};
	}
	template <class T>
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
	template <class T, class... Args>
	Some<T> some(Args&&... args)
	{
		T* ptr = new T(args...);
		return Some<T>::from_raw(ptr);
	}

	template <class T>
	Some<T> empty()
	{
		return Some<T>::from_raw(nullptr);
	}

#ifndef SWIG
	template <class T>
	SG_FORCED_INLINE T wrap(const T& value)
	{
		return value;
	}

	SG_FORCED_INLINE const char* wrap(const char* ptr)
	{
		return ptr;
	}

	template <class T>
	SG_FORCED_INLINE Some<T> wrap(T* ptr)
	{
		return Some<T>::from_raw(ptr);
	}

	template <class T>
	SG_FORCED_INLINE Some<T> wrap(const Some<T>& other)
	{
		return other;
	}
#endif
};

#endif /* __SG_SOME_H__ */
