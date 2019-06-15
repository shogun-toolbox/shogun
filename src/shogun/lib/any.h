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
#include <cassert>
#include <limits>
#include <map>
#include <unordered_map>
#include <stdexcept>
#include <string.h>
#include <string>
#include <typeinfo>
#include <type_traits>
#include <vector>

#include <shogun/util/traits.h>

namespace shogun {

	namespace any_detail
	{
		std::string demangled_type_helper(const char *name);
	}

	/** Converts compiler-dependent name of class to
	 * something human readable.
	 * @return human readable name of class
	 */
	template <typename T>
	std::string demangled_type()
	{
		const char* name = typeid(T).name();

		return any_detail::demangled_type_helper(name);
	}

    template <typename T = void>
	std::string demangled_type(const char* name) {
		return any_detail::demangled_type_helper(name);
	}

	class CSGObject;
	template <class T>
	class SGVector;
	template <class T>
	class SGString;
	template <class T>
	class SGSparseVector;
	template <class T>
	class SGMatrix;
	template <class T>
	class SGSparseMatrix;

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
		void reset(const ArrayReference<T, S>& other);

		S* size() const
		{
			return m_length;
		}
		T** ptr() const
		{
			return m_ptr;
		}
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
		void reset(const Array2DReference<T, S>& other);

		std::pair<S*, S*> size() const
		{
			return std::make_pair(m_rows, m_cols);
		}

		T** ptr() const
		{
			return m_ptr;
		}

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
		virtual void on(char*) = 0;
		virtual void on(int8_t*) = 0;
		virtual void on(uint8_t*) = 0;
		virtual void on(int16_t*) = 0;
		virtual void on(uint16_t*) = 0;
		virtual void on(int32_t*) = 0;
		virtual void on(uint32_t*) = 0;
		virtual void on(int64_t*) = 0;
		virtual void on(uint64_t*) = 0;
		virtual void on(float32_t*) = 0;
		virtual void on(float64_t*) = 0;
		virtual void on(floatmax_t*) = 0;
		virtual void on(complex128_t*) = 0;
		virtual void on(CSGObject**) = 0;
		virtual void enter_matrix(index_t* rows, index_t* cols) = 0;
		virtual void enter_vector(index_t* size) = 0;
		virtual void enter_std_vector(size_t* size) = 0;
		virtual void enter_map(size_t* size) = 0;
		virtual void enter_matrix_row(index_t *rows, index_t *cols) =0;
		virtual void exit_matrix_row(index_t *rows, index_t *cols)=0;
		virtual void exit_matrix(index_t* rows, index_t* cols) = 0;
		virtual void exit_vector(index_t* size) = 0;
		virtual void exit_std_vector(size_t* size) = 0;
		virtual void exit_map(size_t* size) = 0;

		template <typename T>
		void on_matrix_row(index_t* rows, index_t* cols, SGMatrix<T>* _v)
		{
			enter_matrix_row(rows, cols);
			for(index_t i=0; i<*rows; i++)
			{
				on(std::addressof((*_v)(i, *cols)));
			}
			exit_matrix_row(rows, cols);
		}

		template<typename T>
		void on(SGVector<T>* _v)
		{
			auto size = _v->vlen;
			enter_vector(std::addressof(size));
			if (size != _v->vlen)
				_v->resize_vector(size);
			for (auto& _value: *_v)
				on(std::addressof(_value));
			exit_vector(std::addressof(size));
		}

		template<typename T>
		void on(SGString<T>* _v)
		{
			auto size = _v->slen;
			enter_vector(std::addressof(size));
			if (size != _v->slen)
			{
				if (_v->string)
					_v->destroy_string();
				_v->string = SG_MALLOC(T, size);
				_v->slen = size;
			}
			for (index_t i = 0; i < size; ++i)
				on(std::addressof(_v->string[i]));
			exit_vector(std::addressof(size));
		}

		template<typename T>
		void on(SGSparseVector<T>* _v)
		{
			auto size = _v->num_feat_entries*2;
			enter_vector(std::addressof(size));
			assert(size % 2 == 0);
			size /= 2;
			if (size != _v->num_feat_entries)
				*_v = SGSparseVector<T>(size);
			for (index_t i = 0; i < size; ++i)
			{
				on(std::addressof(_v->features[i].feat_index));
				on(std::addressof(_v->features[i].entry));
			}
			exit_vector(std::addressof(size));
		}

		template<typename T>
		void on(SGSparseMatrix<T>* _m)
		{
			//FIXME
			//_m->num_vectors;
			//_m->num_features;
			//_m->sparse_matrix;
		}


		template<class T, class S>
		void on(ArrayReference<T,S>* _v)
		{
			auto size = *(_v->size());
			enter_vector(std::addressof(size));
			if (size != *(_v->size()))
			{
				*_v->size() = size;
				if (*_v->ptr() != nullptr)
					SG_FREE(*_v->ptr());
				if (size)
					*_v->ptr() = SG_CALLOC(T, size);
			}
			auto ptr = *(_v->ptr());
			for (S i = 0; i < size; ++i)
				on(std::addressof(ptr[i]));
			exit_vector(std::addressof(size));
		}

		template<class T, class S>
		void on(Array2DReference<T,S>* _v)
		{
			auto shape = _v->size();
			S rows = *shape.first;
			S cols = *shape.second;
			enter_matrix(shape.first, shape.second);
			int64_t length = ((int64_t)*shape.first)*(*shape.second);
			if ((rows != *shape.first) || (cols != *shape.second))
			{
				if (*_v->ptr() != nullptr)
					SG_FREE(*_v->ptr());
				if (length)
					*_v->ptr() = SG_MALLOC(T, length);
			}
			auto ptr = *(_v->ptr());
			for (int64_t i = 0; i < length; ++i)
				on(std::addressof(ptr[i]));
			exit_matrix(shape.first, shape.second);
		}

		template<typename T>
		void on(SGMatrix<T>* _matrix)
		{
			auto rows = _matrix->num_rows;
			auto cols = _matrix->num_cols;
			enter_matrix(std::addressof(rows), std::addressof(cols));
			if ((rows != _matrix->num_rows) || (cols != _matrix->num_cols))
				*_matrix = SGMatrix<T>(rows, cols);
			for (auto index=0; index < cols; index++) {
				on_matrix_row(std::addressof(rows), std::addressof(index), _matrix);
			}
			exit_matrix(std::addressof(rows), std::addressof(cols));
		}

		template<class T>
		void on(std::vector<T>* _v)
		{
			auto size = _v->size();
			enter_std_vector(std::addressof(size));
			if (size != _v->size())
				_v->resize(size);
			for (auto& _value: *_v)
				on(std::addressof(_value));
			exit_std_vector(std::addressof(size));
		}

		template<class T1, class T2>
		void on(std::map<T1, T2>* _v)
		{
			auto size = _v->size();
			enter_map(std::addressof(size));
			if (size != _v->size())
			{
				// reading
				_v->clear();
				for (size_t i = 0; i < size; ++i)
				{
					std::pair<T1, T2> p;
					on(std::addressof(p.first));
					on(std::addressof(p.second));
					_v->emplace(p);
				}
			}
			else
			{
				// writing
				for (auto _value: *_v)
				{
					on(std::addressof(_value.first));
					on(std::addressof(_value.second));
				}
			}
			exit_map(std::addressof(size));
		}

		template<class T, std::enable_if_t<std::is_base_of<CSGObject, T>::value, T>* = nullptr>
		void on(T** v)
		{
			on((CSGObject**)v);
		}

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
		bool compare_impl_eq(const T& lhs, const T& rhs) = delete;
		template <>
		bool compare_impl_eq(const float32_t& lhs, const float32_t& rhs);
		template <>
		bool compare_impl_eq(const float64_t& lhs, const float64_t& rhs);
		template <>
		bool compare_impl_eq(const floatmax_t& lhs, const floatmax_t& rhs);
		template <>
		bool compare_impl_eq(const complex128_t& lhs, const complex128_t& rhs);

		template<typename T, typename _ = void>
		struct has_special_compare : std::false_type {};

		template<typename T>
		struct has_special_compare<
			T,
			traits::when_exists<
				decltype(compare_impl_eq(std::declval<T>(), std::declval<T>()))
			>
		> : public std::true_type {};

		template <class T>
		bool compare(const T& lhs, const T& rhs)
		{
			if constexpr (traits::has_equals_ptr<T>::value)
			{
				if (lhs && rhs)
					return lhs->equals(rhs);
				else if (!lhs && !rhs)
					return true;
				else
					return false;
			}
			else if constexpr (traits::has_equals<T>::value)
			{
				return lhs.equals(rhs);
			}
			else if constexpr (has_special_compare<T>::value)
			{
				return compare_impl_eq(lhs, rhs);
			}
			else if constexpr (traits::is_comparable<T>::value)
			{
				return (lhs == rhs);
			}
			else if constexpr (traits::is_container<T>::value)
			{
				if (lhs.size() != rhs.size())
				{
					return false;
				}
				for (auto l = lhs.cbegin(), r = rhs.cbegin(); l != lhs.cend(); ++l, ++r)
				{
					if (!compare(*l, *r))
					{
						return false;
					}
				}
			}
			else if constexpr (traits::is_pair<T>::value)
			{
				return compare(lhs.first, rhs.first) && compare(lhs.second, rhs.second);
			}
			else if constexpr (traits::is_functional<T>::value)
			{
				if constexpr (!traits::returns_void<T>::value)
				{
					return compare(lhs(), rhs());
				}
				else
				{
					return false;
				}
			}
			else if constexpr (std::is_same<T, Empty>::value)
			{
				return true;
			}
			else
			{
				// we assert something that is false to see the type T
				static_assert(std::is_same<T, Empty>::value, "Comparison is not supported");
			}
		}

		template <class T>
		size_t hash(const T& value)
		{
			if constexpr (traits::is_hashable<T>::value)
			{
				return std::hash<T>{}(value);
			}
			else if constexpr (traits::is_container<T>::value)
			{
				size_t result = 0;
				for (const auto& it: value) {
					result ^= hash(it);
				}
				return result;
			}
			else
			{
				return 0;
			}
		}

		template <class T, std::enable_if_t<std::is_copy_constructible<T>::value>* = nullptr>
		inline T clone_impl(general, T& value)
		{
			return T(value);
		}

		template<class T1, class T2>
		inline auto clone_impl(general, const std::pair<T1, T2>& value)
		{
			return std::make_pair(
				clone_impl(maybe_most_important(), value.first),
				clone_impl(maybe_most_important(), value.second));
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
		inline T& mutable_value_of(void** ptr)
		{
			return *static_cast<T*>(*ptr);
		}

		template <class T>
		inline T const* typed_pointer(const void* ptr)
		{
			return static_cast<T const*>(ptr);
		}

		template <class T>
		inline auto clone(void** storage, T& value)
		    -> decltype(clone_impl(maybe_most_important(), value))
		{
			auto cloned = clone_impl(maybe_most_important(), value);
			mutable_value_of<decltype(cloned)>(storage) = cloned;
			return cloned;
		}

		template <class T,
			std::enable_if_t<traits::is_container<T>::value>* = nullptr>
		inline auto clone(void** storage, const T& value)
		{
			T cloned;
			std::transform(
				value.cbegin(), value.cend(),
				std::inserter(cloned, cloned.end()),
				[](auto o) {
					return static_cast<typename T::value_type>(
						clone_impl(maybe_most_important(), o));
				});
			mutable_value_of<decltype(cloned)>(storage) = cloned;
		}

		template <class T, class S>
		inline auto clone(void** storage, const ArrayReference<T, S>& value)
		{
			auto existing = mutable_value_of<ArrayReference<T, S>>(storage);
			existing.reset(value);
		}

		template <class T, class S>
		inline auto clone(void** storage, const Array2DReference<T, S>& value)
		{
			auto existing = mutable_value_of<Array2DReference<T, S>>(storage);
			existing.reset(value);
		}

		template <class T>
		inline const T& value_of(T const* ptr)
		{
			if (!ptr)
			{
				throw std::logic_error("Tried to access null pointer");
			}
			return *ptr;
		}

		template <>
		inline const Empty& value_of(Empty const* ptr)
		{
			static Empty empty;
			return empty;
		}

		template <class T>
		T get_value(const void* storage, bool is_functional)
		{
			if (is_functional)
			{
				auto function = typed_pointer<std::function<T()>>(storage);
				return (*function)();
			}
			else
			{
				return value_of(typed_pointer<T>(storage));
			}
		}

		template <class T, class S>
		inline auto free_array(T* ptr, S size)
		{
			if (!ptr)
			{
				return 0;
			}
			SG_FREE(ptr);
			return 0;
		}

		void free_object(CSGObject* obj);

		template <class T, class S>
		inline auto free_array(T** ptr, S size) -> decltype(ptr[0]->unref())
		{
			if (!ptr)
			{
				return 0;
			}
			for (S i = 0; i < size; ++i)
			{
				free_object(ptr[i]);
			}

			SG_FREE(ptr);
			return 0;
		}

		template <class T>
		inline void copy_array(T* begin, T* end, T* dst)
		{
			std::transform(begin, end, dst, [](const T& value) {
				return static_cast<T>(
				    clone_impl(maybe_most_important(), value));
			});
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
	void ArrayReference<T, S>::reset(const ArrayReference<T, S>& other)
	{
		auto src = *(other.m_ptr);
		auto len = *(other.m_length);
		auto& dst = *(this->m_ptr);
		auto own_len = *(this->m_length);
		any_detail::free_array(dst, own_len);
		dst = new T[len];
		*(this->m_length) = len;
		any_detail::copy_array(src, src + len, dst);
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

	template <class T, class S>
	void Array2DReference<T, S>::reset(const Array2DReference<T, S>& other)
	{
		auto src = *(other.m_ptr);
		auto rows = *(other.m_rows);
		auto cols = *(other.m_cols);
		auto& dst = *(this->m_ptr);
		auto own_rows = *(this->m_rows);
		auto own_cols = *(this->m_cols);
		any_detail::free_array(dst, ((long)own_rows * own_cols));
		dst = new T[(long)rows * cols];
		*(this->m_rows) = rows;
		*(this->m_cols) = cols;
		any_detail::copy_array(src, src + ((long)rows * cols), dst);
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

		virtual bool should_inherit_storage() const = 0;

		/** Clones value provided by from into storage
		 * @param storage pointer to a pointer to storage
		 * @param from pointer to value to clone
		 */
		virtual void clone(void** storage, const void* from) const = 0;

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
		virtual bool matches_type(const std::type_info& ti) const = 0;

		/** Checks if policies are compatible.
		 * @param other other policy
		 * @return true if policies do match
		 */
		virtual bool matches_policy(const BaseAnyPolicy* other) const = 0;

		/** Compares two storages.
		 * @param storage pointer to a pointer to storage
		 * @param other_storage pointer to a pointer to another storage
		 * @return true if both storages have same value
		 */
		virtual bool
		equals(const void* storage, const void* other_storage) const = 0;

		/** Visitor pattern. Calls the appropriate 'on' method of AnyVisitor.
		 *
		 * @param storage pointer to storage
		 * @param visitor abstract visitor to use
		 */
		virtual void visit(void* storage, AnyVisitor* visitor) const = 0;

		virtual bool is_functional() const = 0;

		virtual size_t hash(void* storage) const = 0;
	};

	template <typename T>
	class TypedAnyPolicy : public BaseAnyPolicy
	{
	public:
		/** Returns type-name as string.
		 * @return name of type class
		 */
		virtual std::string type() const override
		{
			return demangled_type<T>();
		}

		/** Returns type info
		 * @return type info of value's type
		 */
		virtual const std::type_info& type_info() const override
		{
			return typeid(T);
		}

		/** Compares type.
		 * @param ti type information
		 * @return true if type matches
		 */
		virtual bool matches_type(const std::type_info& ti) const override
		{
			return typeid(T) == ti;
		}

		virtual bool is_functional() const override
		{
			return traits::is_functional<T>::value;
		}
	};

	/** @brief This is one concrete implementation of policy that
	 * uses void pointers to store values.
	 */
	template <typename T>
	class PointerValueAnyPolicy : public TypedAnyPolicy<T>
	{
	public:
		/** Puts provided value pointed by v (untyped to be generic) to storage.
		 * @param storage pointer to a pointer to storage
		 * @param v pointer to value
		 */
		virtual void set(void** storage, const void* v) const override
		{
			*(storage) = new T(value_of(typed_pointer<T>(v)));
		}

		virtual bool should_inherit_storage() const override
		{
			return false;
		}

		/** Clones value provided by from into storage
		 * @param storage pointer to a pointer to storage
		 * @param from pointer to value to clone
		 */
		virtual void clone(void** storage, const void* from) const override
		{
			any_detail::clone(storage, value_of(typed_pointer<T>(from)));
		}

		/** Clears storage.
		 * @param storage pointer to a pointer to storage
		 */
		virtual void clear(void** storage) const override
		{
			delete typed_pointer<T>(*storage);
		}

		/** Checks if policies are compatible.
		 * @param other other policy
		 * @return true if policies do match
		 */
		virtual bool matches_policy(const BaseAnyPolicy* other) const override;

		/** Compares two storages.
		 * @param storage pointer to a pointer to storage
		 * @param other_storage pointer to a pointer to another storage
		 * @return true if both storages have same value
		 */
		bool
		equals(const void* storage, const void* other_storage) const override
		{
			const T& typed_storage = value_of(typed_pointer<T>(storage));
			const T& typed_other_storage =
			    value_of(typed_pointer<T>(other_storage));
			return compare(typed_storage, typed_other_storage);
		}

		/** Visitor pattern. Calls the appropriate 'on' method of AnyVisitor.
		 *
		 * @param storage pointer to a pointer to storage
		 * @param visitor abstract visitor to use
		 */
		virtual void visit(void* storage, AnyVisitor* visitor) const override
		{
			visitor->on(static_cast<T*>(storage));
		}

		virtual size_t hash(void *storage) const override {
			return any_detail::hash(value_of(typed_pointer<T>(storage)));
		}
	};

	template <typename T>
	class NonOwningAnyPolicy : public TypedAnyPolicy<T>
	{
	public:
		/** Puts provided value pointed by v (untyped to be generic) to storage.
		 * @param storage pointer to a pointer to storage
		 * @param v pointer to value
		 */
		virtual void set(void** storage, const void* v) const override
		{
			mutable_value_of<T>(storage) = value_of(typed_pointer<T>(v));
		}

		virtual bool should_inherit_storage() const override
		{
			return true;
		}

		/** Clones value provided by from into storage
		 * @param storage pointer to a pointer to storage
		 * @param from pointer to value to clone
		 */
		virtual void clone(void** storage, const void* from) const override
		{
			any_detail::clone(storage, value_of(typed_pointer<T>(from)));
		}

		/** Clears storage.
		 * @param storage pointer to a pointer to storage
		 */
		virtual void clear(void** storage) const override
		{
		}

		/** Checks if policies are compatible.
		 * @param other other policy
		 * @return true if policies do match
		 */
		virtual bool matches_policy(const BaseAnyPolicy* other) const override;

		/** Compares two storages.
		 * @param storage pointer to a pointer to storage
		 * @param other_storage pointer to a pointer to another storage
		 * @return true if both storages have same value
		 */
		bool
		equals(const void* storage, const void* other_storage) const override
		{
			const T& typed_storage = value_of(typed_pointer<T>(storage));
			const T& typed_other_storage =
			    value_of(typed_pointer<T>(other_storage));
			return compare(typed_storage, typed_other_storage);
		}

		/** Visitor pattern. Calls the appropriate 'on' method of AnyVisitor.
		 *
		 * @param storage pointer to storage
		 * @param visitor abstract visitor to use
		 */
		virtual void visit(void* storage, AnyVisitor* visitor) const override
		{
			visitor->on(static_cast<T*>(storage));
		}

		virtual size_t hash(void* storage) const override {
			return any_detail::hash(value_of(typed_pointer<T>(storage)));
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
	bool NonOwningAnyPolicy<T>::matches_policy(const BaseAnyPolicy* other) const
	{
		if (this == other)
		{
			return true;
		}
		if (other == owning_policy<T>())
		{
			return true;
		}
		return this->matches_type(other->type_info());
	}

	template <class T>
	bool PointerValueAnyPolicy<T>::matches_policy(const BaseAnyPolicy* other) const
	{
		if (this == other)
		{
			return true;
		}
		if (other == non_owning_policy<T>())
		{
			return true;
		}
		return this->matches_type(other->type_info());
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
		Any(const BaseAnyPolicy* the_policy, void* the_storage)
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
			if (!policy->matches_policy(other.policy))
			{
				throw TypeMismatchException(
				    other.policy->type(), policy->type());
			}
			policy->clear(&storage);
			if (other.policy->should_inherit_storage())
			{
				policy = other.policy;
			}
			set_or_inherit(other);
			return *(this);
		}

		Any& clone_from(const Any& other)
		{
			if (!other.cloneable())
			{
				throw std::logic_error("Tried to clone non-cloneable Any");
			}
			if (empty())
			{
				policy = other.policy;
				set_or_inherit(other);
				return *(this);
			}
			if (!policy->matches_policy(other.policy))
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
		T as() const
		{
			if (has_type<T>())
			{
				return any_detail::get_value<T>(
				    storage, policy->is_functional());
			}
			else
			{
				throw TypeMismatchException(
				    demangled_type<T>(), policy->type());
			}
		}

		/** @return true if type is same. */
		template <typename T>
		inline bool has_type() const
		{
			return (policy == owning_policy<T>()) ||
			       (policy == non_owning_policy<T>()) ||
			       policy == owning_policy<std::function<T()>>() ||
			       has_type_fallback<T>();
		}

		/** @return true if type-id is same. */
		template <typename T>
		bool has_type_fallback() const
		{
			return policy->matches_type(typeid(T)) ||
			       policy->matches_type(typeid(std::function<T()>));
		}

		/** @return true if Any object is empty. */
		bool empty() const
		{
			return has_type<Empty>();
		}

		/** @return true if Any object is cloneable. */
		bool cloneable() const
		{
			return !policy->is_functional();
		}

		/** @return true if Any object is visitable. */
		bool visitable() const
		{
			return !policy->is_functional();
		}

		/** @return true if Any object is hashable. */
		bool hashable() const
		{
			return !policy->is_functional();
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

		/** Hashes the underlying value, shallow. Uses std::hash.
		 *
		 * @return the value of hash function or 0 if hashing is not supported.
		 */
		size_t hash() const
		{
			return policy->hash(storage);
		}

		/** Visitor pattern. Calls the appropriate 'on' method of AnyVisitor.
		 *
		 * @param visitor visitor object to use
		 */
		void visit(AnyVisitor* visitor) const
		{
			if (!visitable())
			{
				throw std::logic_error("Tried to visit non-visitable Any");
			}
			policy->visit(storage, visitor);
		}
	private:
		void set_or_inherit(const Any& other)
		{
			if (other.policy->should_inherit_storage())
			{
				storage = other.storage;
			}
			else
			{
				policy->set(&storage, other.storage);
			}
		}

	private:
		const BaseAnyPolicy* policy;
		void* storage;
	};

	inline bool operator==(const Any& lhs, const Any& rhs)
	{
		if (lhs.empty() || rhs.empty())
		{
			return lhs.empty() && rhs.empty();
		}
		if (!lhs.policy->matches_policy(rhs.policy))
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

	/** Wraps a function as as instance of Any.
	 *
	 * @param func function to wrap
	 * @return Any object that uses the function to obtain its value
	 */
	template <typename T>
	inline Any make_any(std::function<T()> func)
	{
		return Any(func);
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
