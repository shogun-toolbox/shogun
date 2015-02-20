#ifndef __SG_MAYBE_H__
#define __SG_MAYBE_H__

#ifdef HAVE_CXX11

#include <shogun/lib/ShogunException.h>

namespace shogun
{

	/** @class Holder that represents an object that can be 
	 * either present or absent. Quite simllar to std::optional
	 * introduced in C++14 but provides a way to pass the reason 
	 * of absence (e.g. "incorrect parameter").
	 */
	template <typename T>
	class Maybe
	{
		public:
			Maybe(const T& val) :
				value(val),
				absenceReason(nullptr)
			{
			}
			Maybe(const Maybe& other) :
				value(other.value),
				absenceReason(other.absenceReason)
			{
			}

			/** Returns instance of the class that correspond
			 * to absent value. 
			 *
			 * @param reason a reason why the object is absent
			 *
			 */
			static Maybe<T> nope(const char* reason="Unknown")
			{
				return Maybe(reason);
			}

			/** Evaluates to true when object is present, false
			 * otherwise.
			 *
			 */
			inline operator bool() const
			{
				return absenceReason != nullptr;
			}
			
			/** Returns instance if it is present, fails otherwise
			 */
			inline operator T() const
			{
				if (*this)
					return value;
				else
					throw ShogunException("Tried to access absent object");
			}
			/** Provides a way to call member functions and access
			 * members of the object if it is present, fails otherwise
			 */
			inline T* operator->()
			{
				if (*this)
					return &value;
				else
					throw ShogunException("Tried to access absent object");
			}
		private:
			Maybe();
			Maybe(const char* reason) :
				value(),
				absenceReason(reason)
			{
			}

		private:
			T value;
			const char* absenceReason;
	};

}
#endif
#endif
