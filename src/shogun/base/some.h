#ifndef __SG_SOME_H__
#define __SG_SOME_H__

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
	 * Note: Due to SG_REF/SG_UNREF used in Shogun now, it also imitates 
	 * Shogun references so that shared_ptr counter should always
	 * be equal to the Shogun's object reference count. This will stay
	 * until SG_REF/SG_UNREF are gone.
	 *
	 */
	template <typename T>
	class Some : protected std::shared_ptr<T>
	{
		public:
			Some() = delete;
			Some(const std::shared_ptr<T>& shared);
			Some(const Some<T>& other);
			Some(Some<T>&& other);
			Some& operator=(const Some<T>& other);
			~Some();

			/** Casts the underlying object back to raw pointer 
			 *
			 * @return raw pointer (with SG_REF)
			 */
			operator T*();
			/** Call member function or access member of T 
			 *
			 * @return raw pointer (without SG_REF)
			 */
			T* operator->();
		private:
			using std::shared_ptr<T>::get;
	};

	template <typename T>
	Some<T>::Some(const std::shared_ptr<T>& shared)
		: std::shared_ptr<T>(shared)
	{
	}
	template <typename T>
	Some<T>::Some(const Some<T>& other)
		: std::shared_ptr<T>(other)
	{
	}
	template <typename T>
	Some<T>::Some(Some<T>&& other)
		: std::shared_ptr<T>(other)
	{
	}
	template <typename T>
	Some<T>::~Some()
	{
	}
	template <typename T>
	Some<T>::operator T*()
	{
		T* ptr = this->get();
		SG_REF(ptr);
		return ptr;
	}
	template <typename T>
	T* Some<T>::operator->()
	{
		T* ptr = this->get();
		return ptr;
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
		SG_REF(ptr);
		return std::shared_ptr<T>(ptr, [](T* p) { SG_UNREF(p); });
	}

};

#endif /* HAVE_CXX11 */
#endif /* __SG_SOME_H__ */
