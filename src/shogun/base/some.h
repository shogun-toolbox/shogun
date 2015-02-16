#ifndef __SG_SOME_H__
#define __SG_SOME_H__

#include <memory>
#include <shogun/base/SGObject.h>
#include <shogun/lib/RefCount.h>

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
	class Some
	{
		void* shared;

		public:
			Some(const Some<T>& other);
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

			/** Returns empty instance. For internal use
			 *
			 */
			static inline Some<T> empty()
			{
				return Some();
			}
			/** Returns raw pointer to shared instance. For internal use
			 *
			 */
			static inline void* raw(Some<T> s)
			{
				return s.shared;
			}

		private:
			Some();

			int ref() const;
			int unref() const;

		private:
			void assign(const Some<T>& other);
	};

	template <typename T>
	struct Shared
	{
		RefCount rc;
		T object;

		static inline Shared* from(void* ptr)
		{
			return static_cast<Shared*>(ptr);
		}
	};
	
	template <typename T>
	void Some<T>::assign(const Some<T>& other)
	{
		this->unref();
		other.ref();
		this->shared = other.shared;
	}

	template <typename T>
	Some<T>::Some()
	{
		shared = malloc(sizeof(Shared<T>));
		new (&Shared<T>::from(shared)->rc) RefCount();
	}

	template <typename T>
	int Some<T>::ref() const
	{
		int rc = Shared<T>::from(shared)->rc.ref();
#ifndef TRANSITION_TO_SOME_DONE
		int objrc = Shared<T>::from(shared)->object.ref();
		if (rc != objrc)
			throw std::exception();
#endif
		return rc;
	}

	template <typename T>
	int Some<T>::unref() const
	{
		int rc = Shared<T>::from(shared)->rc.unref();
#ifndef TRANSITION_TO_SOME_DONE
		int objrc = Shared<T>::from(shared)->object.unref();
		if (rc != objrc)
			throw std::exception();
#endif
		if (!rc)
			free(Shared<T>::from(shared));
		return rc;
	}

	template <typename T>
	Some<T>::Some(const Some<T>& other)
	{
		assign(other);
	}

	template <typename T>
	Some<T>& Some<T>::operator=(const Some<T>& other)
	{
		assign(other);
		return *this;
	}

	template <typename T>
	Some<T>::~Some()
	{
		this->unref();
	}
	template <typename T>
	Some<T>::operator T*()
	{
		this->ref();
		return &(Shared<T>::from(shared)->object);
	}
	template <typename T>
	T* Some<T>::operator->()
	{
		return &(Shared<T>::from(shared)->object);
	}

#define __PARAMETERS_0
#define __PARAMETERS_1(UM) UM
#define __PARAMETERS_2(UM, DOIS) UM, DOIS
#define __PARAMETERS_3(UM, DOIS, TRES) UM, DOIS, TRES
#define __PARAMETERS_4(UM, DOIS, TRES, QUATRO) UM, DOIS, TRES, QUATRO
#define __PARAMETERS_5(UM, DOIS, TRES, QUATRO, CINCO) UM, DOIS, TRES, QUATRO, CINCO
#define __CREATE_SOME(X) \
	Some<T> s = Some<T>::empty(); \
	T* ptr = &(Shared<T>::from(Some<T>::raw(s))->object); \
	new (ptr) T(X); \
	return s

	template <typename T>
	inline Some<T> some()
	{
		__CREATE_SOME(__PARAMETERS_0);
	}
	template <typename T, typename Um>
	inline Some<T> some(const Um& um)
	{
		__CREATE_SOME(__PARAMETERS_1(um));
	}
	template <typename T, typename Um, typename Dois>
	inline Some<T> some(const Um& um, const Dois& dois)
	{
		__CREATE_SOME(__PARAMETERS_2(um, dois));
	}
	template <typename T, typename Um, typename Dois, typename Tres>
	inline Some<T> some(const Um& um, const Dois& dois, const Tres& tres)
	{
		__CREATE_SOME(__PARAMETERS_3(um, dois, tres));
	}
	template <typename T, typename Um, typename Dois, typename Tres, typename Quatro>
	inline Some<T> some(const Um& um, const Dois& dois, const Tres& tres, const Quatro& quatro)
	{
		__CREATE_SOME(__PARAMETERS_4(um, dois, tres, quatro));
	}
	template <typename T, typename Um, typename Dois, typename Tres, typename Quatro, typename Cinco>
	inline Some<T> some(const Um& um, const Dois& dois, const Tres& tres, const Quatro& quatro, const Cinco& cinco)
	{
		__CREATE_SOME(__PARAMETERS_5(um, dois, tres, quatro, cinco));
	}
#undef __PARAMETERS_0
#undef __PARAMETERS_1
#undef __PARAMETERS_2
#undef __PARAMETERS_3
#undef __PARAMETERS_4
#undef __PARAMETERS_5
#undef __CREATE_SOME

};
#endif /* __SG_SOME_H__ */
