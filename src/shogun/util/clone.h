/*
* Written (W) 2019 Giovanni De Toni
*/

#include <type_traits>

#ifndef SHOGUN_CLONE_H
#define SHOGUN_CLONE_H

namespace shogun {
	namespace clone_utils
	{

		struct clone_by_cctor
		{
		};

		struct clone_from_value : clone_by_cctor
		{
		};

		struct clone_from_pointer : clone_from_value
		{
		};

		template <class T, std::enable_if_t<std::is_copy_constructible<T>::value>* = nullptr>
		inline T clone_impl(clone_by_cctor, const T& value)
		{
			return T(value);
		}

		template <class T>
		inline auto clone_impl(clone_from_value, T& value)
		-> decltype(value.clone())
		{
			return value.clone();
		}

		template <class T>
		inline auto clone_impl(clone_from_pointer, T* value)
		-> decltype(value->clone())
		{
			return value->clone();
		}

		/**
		 * Clone a Shogun object by calling its clone() method.
		 * It also works with non-Shogun object by using the copy
		 * constructor.
		 * @tparam T type
		 * @param value value we want to clone
		 * @return the cloned value
		 */
		template <class T>
		inline auto clone(T& value)
		-> decltype(clone_impl(clone_from_pointer(), value))
		{
			return clone_impl(clone_from_pointer(), value);
		}
	}
}

#endif //SHOGUN_CLONE_H
