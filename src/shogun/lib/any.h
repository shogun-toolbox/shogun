/*
 * Copyright (c) 2016, Shogun-Toolbox e.V. <shogun-team@shogun-toolbox.org>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Written (W) 2016 Sergey Lisitsyn
 * Written (W) 2016 Sanuj Sharma
 */

#ifndef _ANY_H_
#define _ANY_H_

#include <shogun/base/init.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string.h>
#include <string>
#include <typeinfo>
#ifdef HAVE_CXA_DEMANGLE
#include <cxxabi.h>
#endif

namespace shogun
{
	/** Converts compiler-dependent name of class to
	 * something human readable.
	 * @return human readable name of class
	 */
	template <typename T>
	std::string demangled_type()
	{
#ifdef HAVE_CXA_DEMANGLE
		size_t length;
		int status;
		char* demangled =
		    abi::__cxa_demangle(typeid(T).name(), nullptr, &length, &status);
		std::string demangled_string(demangled);
		free(demangled);
#else
		std::string demangled_string(typeid(T).name());
#endif
		return demangled_string;
	}

	enum class PolicyType
	{
		OWNING,
		NON_OWNING
	};

	class CSGObject;
	template <class T>
	class SGVector;
	template <class T>
	class SGMatrix;

	class TypeMismatchException : public std::exception
	{
	public:
		TypeMismatchException(
		    const std::string& expected, const std::string& actual)
		    : m_expected(expected), m_actual(actual)
		{
		}
		TypeMismatchException(const TypeMismatchException& other)
		    : m_expected(other.m_expected), m_actual(other.m_actual)
		{
		}
		std::string expected() const
		{
			return m_expected;
		}
		std::string actual() const
		{
			return m_actual;
		}

	private:
		std::string m_expected;
		std::string m_actual;
	};

	template <class T, class S>
	class ArrayReference
	{
	public:
		ArrayReference(T** ptr, S* length) : m_ptr(ptr), m_length(length)
		{
		}
		ArrayReference(const ArrayReference<T, S>& other)
		    : m_ptr(other.m_ptr), m_length(other.m_length)
		{
		}
		ArrayReference<T, S> operator=(const ArrayReference<T, S>& other)
		{
			throw std::logic_error("Assignment not supported");
		}
		bool equals(const ArrayReference<T, S>& other) const;

	private:
		T** m_ptr;
		S* m_length;
	};

	template <class T, class S>
	class Array2DReference
	{
	public:
		Array2DReference(T** ptr, S* rows, S* cols)
		    : m_ptr(ptr), m_rows(rows), m_cols(cols)
		{
		}
		Array2DReference(const Array2DReference<T, S>& other)
		    : m_ptr(other.m_ptr), m_rows(other.m_rows), m_cols(other.m_cols)
		{
		}
		Array2DReference<T, S> operator=(const Array2DReference<T, S>& other)
		{
			throw std::logic_error("Assignment not supported");
		}
		bool equals(const Array2DReference<T, S>& other) const;

	private:
		T** m_ptr;
		S* m_rows;
		S* m_cols;
	};

	/** Used to denote an empty Any object */
	struct Empty
	{
		/** Equality operator */
		bool operator==(const Empty& other) const
		{
			return true;
		}
	};

	class AnyVisitor
	{
	public:
		virtual ~AnyVisitor() = default;

		virtual void on(bool*) = 0;
		virtual void on(int32_t*) = 0;
		virtual void on(int64_t*) = 0;
		virtual void on(float*) = 0;
		virtual void on(double*) = 0;
		virtual void on(CSGObject**) = 0;
		virtual void on(SGVector<int>*) = 0;
		virtual void on(SGVector<float>*) = 0;
		virtual void on(SGVector<double>*) = 0;
		virtual void on(SGMatrix<int>*) = 0;
		virtual void on(SGMatrix<float>*) = 0;
		virtual void on(SGMatrix<double>*) = 0;

		void on(Empty*)
		{
		}

		void on(...)
		{
		}
	};

	namespace any_detail
	{

		struct by_default
		{
		};

		struct general : by_default
		{
		};

		struct more_important : general
		{
		};

		struct maybe_most_important : more_important
		{
		};

		template <class T>
		auto compare_impl(by_default, const T& lhs, const T& rhs) = delete;

		template <class T>
		bool compare_impl_eq(const T& lhs, const T& rhs)
		{
			return lhs == rhs;
		}
		template <>
		bool compare_impl_eq(const float32_t& lhs, const float32_t& rhs);
		template <>
		bool compare_impl_eq(const float64_t& lhs, const float64_t& rhs);
		template <>
		bool compare_impl_eq(const floatmax_t& lhs, const floatmax_t& rhs);
		template <>
		bool compare_impl_eq(const complex128_t& lhs, const complex128_t& rhs);

		template <class T>
		auto compare_impl(general, const T& lhs, const T& rhs)
		    -> decltype(lhs == rhs)
		{
			return compare_impl_eq(lhs, rhs);
		}

		template <class T>
		auto compare_impl(more_important, const T& lhs, const T& rhs)
		    -> decltype(lhs.equals(rhs))
		{
			return lhs.equals(rhs);
		}

		template <class T>
		auto compare_impl(maybe_most_important, T* lhs, T* rhs)
		    -> decltype(lhs->equals(rhs))
		{
			if (lhs && rhs)
				return lhs->equals(rhs);
			else if (!lhs && !rhs)
				return true;
			else
				return false;
		}

		template <class T>
		inline bool compare(const T& lhs, const T& rhs)
		{
			return compare_impl(maybe_most_important(), lhs, rhs);
		}

		template <class T>
		inline T clone_impl(general, T& value)
		{
			return value;
		}

		template <class T>
		inline auto clone_impl(more_important, const T& value)
		    -> decltype(value.clone())
		{
			return value.clone();
		}

		template <class T>
		inline auto clone_impl(maybe_most_important, T* value)
		    -> decltype(static_cast<void*>(value->clone()))
		{
			if (!value)
				return nullptr;

			return static_cast<void*>(value->clone());
		}

		template <class T>
		inline auto clone(T& value)
		    -> decltype(clone_impl(maybe_most_important(), value))
		{
			return clone_impl(maybe_most_important(), value);
		}

		template <class T>
		inline T const* typed_pointer(const void* ptr)
		{
			return static_cast<T const*>(ptr);
		}

		template <class T>
		inline const T& value_of(T const* ptr)
		{
			return *ptr;
		}

		template <class T>
		inline T& mutable_value_of(void** ptr)
		{
			return *static_cast<T*>(*ptr);
		}
	}

	using any_detail::typed_pointer;
	using any_detail::value_of;
	using any_detail::mutable_value_of;
	using any_detail::compare;

	template <class T, class S>
	bool ArrayReference<T, S>::equals(const ArrayReference<T, S>& other) const
	{
		if (*(m_length) != *(other.m_length))
		{
			return false;
		}
		if (*(m_ptr) == *(other.m_ptr))
		{
			return true;
		}
		return std::equal(
		    *(m_ptr), *(m_ptr) + *(m_length), *(other.m_ptr),
		    [](T lhs, T rhs) -> bool { return any_detail::compare(lhs, rhs); });
	}

	template <class T, class S>
	bool
	Array2DReference<T, S>::equals(const Array2DReference<T, S>& other) const
	{
		if ((*(m_rows) != *(other.m_rows)) || (*(m_cols) != *(other.m_cols)))
		{
			return false;
		}
		if (*(m_ptr) == *(other.m_ptr))
		{
			return true;
		}
		int64_t size = int64_t(*(m_rows)) * (*(m_cols));
		return std::equal(
		    *(m_ptr), *(m_ptr) + size, *(other.m_ptr),
		    [](T lhs, T rhs) -> bool { return any_detail::compare(lhs, rhs); });
	}

	/** @brief An interface for a policy to store a value.
	 * Value can be any data like primitive data-types, shogun objects, etc.
	 * Policy defines how to handle this data. It works with a
	 * provided memory region and is able to set value, clear it
	 * and return the type-name as string.
	 */
	class BaseAnyPolicy
	{
	public:
		/** Puts provided value pointed by v (untyped to be generic) to storage.
		 * @param storage pointer to a pointer to storage
		 * @param v pointer to value
		 */
		virtual void set(void** storage, const void* v) const = 0;

		/** Clones value provided by v into storage
		 * @param storage pointer to a pointer to storage
		 * @param v pointer to value to clone
		 */
		virtual void clone(void** storage, const void* v) const = 0;

		/** Clears storage.
		 * @param storage pointer to a pointer to storage
		 */
		virtual void clear(void** storage) const = 0;

		/** Returns type-name as string.
		 * @return name of type class
		 */
		virtual std::string type() const = 0;

		/** Returns type info
		 * @return type info of value's type
		 */
		virtual const std::type_info& type_info() const = 0;

		/** Compares type.
		 * @param ti type information
		 * @return true if type matches
		 */
		virtual bool matches(const std::type_info& ti) const = 0;

		/** Checks if policies are compatible.
		 * @param other other policy
		 * @return true if policies do match
		 */
		virtual bool matches(BaseAnyPolicy* other) const = 0;

		/** Compares two storages.
		 * @param storage pointer to a pointer to storage
		 * @param other_storage pointer to a pointer to another storage
		 * @return true if both storages have same value
		 */
		virtual bool
		equals(const void* storage, const void* other_storage) const = 0;

		/** Returns the type of policy.
		 * @return type of policy
		 */
		virtual PolicyType policy_type() const = 0;

		/** Visitor pattern. Calls the appropriate 'on' method of AnyVisitor.
		 *
		 * @param storage pointer to storage
		 * @param visitor abstract visitor to use
		 */
		virtual void visit(void* storage, AnyVisitor* visitor) const = 0;
	};

	/** @brief This is one concrete implementation of policy that
	 * uses void pointers to store values.
	 */
	template <typename T>
	class PointerValueAnyPolicy : public BaseAnyPolicy
	{
	public:
		/** Puts provided value pointed by v (untyped to be generic) to storage.
		 * @param storage pointer to a pointer to storage
		 * @param v pointer to value
		 */
		virtual void set(void** storage, const void* v) const
		{
			*(storage) = new T(value_of(typed_pointer<T>(v)));
		}

		/** Clones value provided by v into storage
		 * @param storage pointer to a pointer to storage
		 * @param v pointer to value to clone
		 */
		virtual void clone(void** storage, const void* v) const
		{
			auto cloned = any_detail::clone(value_of(typed_pointer<T>(v)));
			mutable_value_of<decltype(cloned)>(storage) = cloned;
		}

		/** Clears storage.
		 * @param storage pointer to a pointer to storage
		 */
		virtual void clear(void** storage) const
		{
			delete typed_pointer<T>(*storage);
		}

		/** Returns type-name as string.
		 * @return name of type class
		 */
		virtual std::string type() const
		{
			return demangled_type<T>();
		}

		/** Returns type info
		 * @return type info of value's type
		 */
		virtual const std::type_info& type_info() const
		{
			return typeid(T);
		}

		/** Compares type.
		 * @param ti type information
		 * @return true if type matches
		 */
		virtual bool matches(const std::type_info& ti) const
		{
			return typeid(T) == ti;
		}

		/** Checks if policies are compatible.
		 * @param other other policy
		 * @return true if policies do match
		 */
		virtual bool matches(BaseAnyPolicy* other) const;

		/** Compares two storages.
		 * @param storage pointer to a pointer to storage
		 * @param other_storage pointer to a pointer to another storage
		 * @return true if both storages have same value
		 */
		bool equals(const void* storage, const void* other_storage) const
		{
			const T& typed_storage = value_of(typed_pointer<T>(storage));
			const T& typed_other_storage =
			    value_of(typed_pointer<T>(other_storage));
			return compare(typed_storage, typed_other_storage);
		}

		virtual PolicyType policy_type() const
		{
			return PolicyType::OWNING;
		}

		/** Visitor pattern. Calls the appropriate 'on' method of AnyVisitor.
		 *
		 * @param storage pointer to a pointer to storage
		 * @param visitor abstract visitor to use
		 */
		virtual void visit(void* storage, AnyVisitor* visitor) const
		{
			visitor->on(typed_pointer<T>(storage));
		}
	};

	template <typename T>
	class NonOwningAnyPolicy : public BaseAnyPolicy
	{
	public:
		/** Puts provided value pointed by v (untyped to be generic) to storage.
		 * @param storage pointer to a pointer to storage
		 * @param v pointer to value
		 */
		virtual void set(void** storage, const void* v) const
		{
			mutable_value_of<T>(storage) = value_of(typed_pointer<T>(v));
		}

		/** Clones value provided by v into storage
		 * @param storage pointer to a pointer to storage
		 * @param v pointer to value to clone
		 */
		virtual void clone(void** storage, const void* v) const
		{
			auto cloned = any_detail::clone(value_of(typed_pointer<T>(v)));
			mutable_value_of<decltype(cloned)>(storage) = cloned;
		}

		/** Clears storage.
		 * @param storage pointer to a pointer to storage
		 */
		virtual void clear(void** storage) const
		{
		}

		/** Returns type-name as string.
		 * @return name of type class
		 */
		virtual std::string type() const
		{
			return demangled_type<T>();
		}

		/** Returns type info
		 * @return type info of value's type
		 */
		virtual const std::type_info& type_info() const
		{
			return typeid(T);
		}

		/** Compares type.
		 * @param ti type information
		 * @return true if type matches
		 */
		virtual bool matches(const std::type_info& ti) const
		{
			return typeid(T) == ti;
		}

		/** Checks if policies are compatible.
		 * @param other other policy
		 * @return true if policies do match
		 */
		virtual bool matches(BaseAnyPolicy* other) const;

		/** Compares two storages.
		 * @param storage pointer to a pointer to storage
		 * @param other_storage pointer to a pointer to another storage
		 * @return true if both storages have same value
		 */
		bool equals(const void* storage, const void* other_storage) const
		{
			const T& typed_storage = value_of(typed_pointer<T>(storage));
			const T& typed_other_storage =
			    value_of(typed_pointer<T>(other_storage));
			return compare(typed_storage, typed_other_storage);
		}

		virtual PolicyType policy_type() const
		{
			return PolicyType::NON_OWNING;
		}

		/** Visitor pattern. Calls the appropriate 'on' method of AnyVisitor.
		 *
		 * @param storage pointer to storage
		 * @param visitor abstract visitor to use
		 */
		virtual void visit(void* storage, AnyVisitor* visitor) const
		{
			visitor->on(typed_pointer<T>(storage));
		}
	};

	template <typename T>
	static BaseAnyPolicy* owning_policy()
	{
		typedef PointerValueAnyPolicy<T> Policy;
		static Policy policy;
		return &policy;
	}

	template <typename T>
	static BaseAnyPolicy* non_owning_policy()
	{
		typedef NonOwningAnyPolicy<T> Policy;
		static Policy policy;
		return &policy;
	}

	template <class T>
	bool NonOwningAnyPolicy<T>::matches(BaseAnyPolicy* other) const
	{
		if (this == other)
		{
			return true;
		}
		if (other == owning_policy<T>())
		{
			return true;
		}
		return matches(other->type_info());
	}

	template <class T>
	bool PointerValueAnyPolicy<T>::matches(BaseAnyPolicy* other) const
	{
		if (this == other)
		{
			return true;
		}
		if (other == non_owning_policy<T>())
		{
			return true;
		}
		return matches(other->type_info());
	}

	/** @brief Allows to store objects of arbitrary types
	 * by using a BaseAnyPolicy and provides a type agnostic API.
	 * See its usage in CSGObject::Self, CSGObject::set(), CSGObject::get()
	 * and CSGObject::has().
	 * .
	 */
	class Any
	{
	public:

		/** Empty value constructor */
		Any() : Any(owning_policy<Empty>(), nullptr)
		{
		}

		/** Base constructor */
		Any(BaseAnyPolicy* the_policy, void* the_storage)
		    : policy(the_policy), storage(the_storage)
		{
		}

		/** Constructor to copy value */
		template <typename T>
		explicit Any(const T& v) : Any(owning_policy<T>(), nullptr)
		{
			policy->set(&storage, &v);
		}

		/** Copy constructor */
		Any(const Any& other) : Any(other.policy, nullptr)
		{
			set_or_inherit(other);
		}

		/** Move constructor */
		Any(Any&& other) : Any(other.policy, nullptr)
		{
			set_or_inherit(other);
		}

		/** Assignment operator
		 * @param other another Any object
		 * @return Any object
		 */
		Any& operator=(const Any& other)
		{
			if (empty())
			{
				policy = other.policy;
				set_or_inherit(other);
				return *(this);
			}
			if (!policy->matches(other.policy))
			{
				throw TypeMismatchException(
				    other.policy->type(), policy->type());
			}
			policy->clear(&storage);
			if (other.policy->policy_type() == PolicyType::NON_OWNING)
			{
				policy = other.policy;
				storage = other.storage;
			}
			else
			{
				policy->set(&storage, other.storage);
			}
			return *(this);
		}

		Any& clone_from(const Any& other)
		{
			if (!policy->matches(other.policy))
			{
				throw TypeMismatchException(
				    other.policy->type(), policy->type());
			}
			policy->clone(&storage, other.storage);
			return *(this);
		}

		/** Equality operator
		 * @param lhs Any object on left hand side
		 * @param rhs Any object on right hand side
		 * @return true if both are equal
		 */
		friend inline bool operator==(const Any& lhs, const Any& rhs);

		/** Inequality operator
		 * @param lhs Any object on left hand side
		 * @param rhs Any object on right hand side
		 * @return false if both are equal
		 */
		friend inline bool operator!=(const Any& lhs, const Any& rhs);

		/** Destructor */
		~Any()
		{
			policy->clear(&storage);
		}

		/** Casts hidden value to provided type, fails otherwise.
		 * @return type-casted value
		 */
		template <typename T>
		const T& as() const
		{
			if (same_type<T>())
			{
				return value_of(typed_pointer<T>(storage));
			}
			else
			{
				throw TypeMismatchException(
				    demangled_type<T>(), policy->type());
			}
		}

		/** @return true if type is same. */
		template <typename T>
		inline bool same_type() const
		{
			return (policy == owning_policy<T>()) ||
			       (policy == non_owning_policy<T>()) ||
			       same_type_fallback<T>();
		}

		/** @return true if type-id is same. */
		template <typename T>
		bool same_type_fallback() const
		{
			return policy->matches(typeid(T));
		}

		/** @return true if Any object is empty. */
		bool empty() const
		{
			return same_type<Empty>();
		}

		const std::type_info& type_info() const
		{
			return policy->type_info();
		}

		/** Returns type-name of policy as string.
		 * @return name of type class
		 */
		std::string type() const
		{
			return policy->type();
		}

		/** Visitor pattern. Calls the appropriate 'on' method of AnyVisitor.
		 *
		 * @param visitor visitor object to use
		 */
		void visit(AnyVisitor* visitor) const
		{
			policy->visit(storage, visitor);
		}

	private:
		void set_or_inherit(const Any& other)
		{
			if (other.policy->policy_type() == PolicyType::NON_OWNING)
			{
				storage = other.storage;
			}
			else
			{
				policy->set(&storage, other.storage);
			}
		}

	private:
		BaseAnyPolicy* policy;
		void* storage;
	};

	inline bool operator==(const Any& lhs, const Any& rhs)
	{
		if (lhs.empty() || rhs.empty())
		{
			return lhs.empty() && rhs.empty();
		}
		if (!lhs.policy->matches(rhs.policy))
		{
			return false;
		}
		void* lhs_storage = lhs.storage;
		void* rhs_storage = rhs.storage;
		return lhs.policy->equals(lhs_storage, rhs_storage);
	}

	inline bool operator!=(const Any& lhs, const Any& rhs)
	{
		return !(lhs == rhs);
	}

	/** Erases value type i.e. converts it to Any
	 * For input object of any type, it returns an Any object
	 * which stores the input object's raw value. It saves the type
	 * information internally to be cast back later by using any_cast().
	 *
	 * @param v value
	 * @return Any object with the input value
	 */
	template <typename T>
	inline Any make_any(const T& v)
	{
		return Any(v);
	}

	template <typename T>
	inline Any make_any_ref(T* v)
	{
		return Any(non_owning_policy<T>(), v);
	}

	template <typename T, typename S>
	inline Any make_any_ref(T** ptr, S* length)
	{
		return make_any(ArrayReference<T, S>(ptr, length));
	}

	template <typename T, typename S>
	inline Any make_any_ref(T** ptr, S* rows, S* cols)
	{
		return make_any(Array2DReference<T, S>(ptr, rows, cols));
	}

	/** Tries to recall Any type, fails when type is wrong.
	 * Any stores type information of an object internally in a BaseAnyPolicy.
	 * This function returns type-casted value if the internal type information
	 * matches with the provided typename, otherwise throws std::logic_error.
	 *
	 * @param any object of Any
	 * @return type-casted value
	 */
	template <typename T>
	inline T any_cast(const Any& any)
	{
		return any.as<T>();
	}
}

#endif //_ANY_H_
