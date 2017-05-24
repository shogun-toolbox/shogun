#ifndef __SG_SOME_H__
#define __SG_SOME_H__

#include <memory>

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
		template <typename R>
		Some(const Some<R>& other);
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
		operator T*() const;
		/** Call member function or access member of T
		 *
		 * @return raw pointer (without SG_REF)
		 */
		T* operator->() const;

		/**
		 * Get the raw pointer
		 *
		 * @return raw pointer (without SG_REF)
		 */
		T* get() const;

	private:
		Some();
		void unref();
		void ref();

	private:
		T* raw = nullptr;
	};

	template <typename T>
	Some<T>::Some() : raw(nullptr)
	{
	}
	template <typename T>
	Some<T>::Some(const Some<T>& other) : raw(other.raw)
	{
		ref();
	}
	template <typename T>
	Some<T>::Some(T* other) : raw(other)
	{
		ref();
	}
	template <class T>
	template <class R>
	Some<T>::Some(const Some<R>& other)
	{
		raw = dynamic_cast<T*>(other.get());
		ref();
	}
	template <typename T>
	Some<T>& Some<T>::operator=(T* other)
	{
		if (raw != other)
		{
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
	Some<T>::operator T*() const
	{
		return raw;
	}
	template <typename T>
	T* Some<T>::operator->() const
	{
		return raw;
	}
	template <class T>
	T* Some<T>::get() const
	{
		return raw;
	}
	template <typename T>
	void Some<T>::ref()
	{
		if (raw)
			(raw)->ref();
	}
	template <typename T>
	void Some<T>::unref()
	{
		if (raw)
		{
			if ((raw)->unref() == 0)
				(raw) = NULL;
		};
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

#endif /* __SG_SOME_H__ */
