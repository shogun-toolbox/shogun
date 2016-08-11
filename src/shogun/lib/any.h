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

#include <string.h>
#include <stdexcept>
#include <typeinfo>
#include <cxxabi.h>
#include <shogun/io/SGIO.h>

namespace shogun
{

#ifndef SWIG // SWIG should skip this part
	namespace serial
	{
		enum EnumContainerType
		{
			CT_UNDEFINED,
			CT_PRIMITIVE,
			CT_SGVECTOR,
			CT_SGMATRIX
		};

		enum EnumPrimitiveType
	    {
	        PT_UNDEFINED,
	        PT_BOOL_TYPE,
	        PT_CHAR_TYPE,
	        PT_INT_8,
	        PT_UINT_8,
	        PT_INT_16,
	        PT_UINT_16,
	        PT_INT_32,
	        PT_UINT_32,
	        PT_INT_64,
	        PT_UINT_64,
	        PT_FLOAT_32,
	        PT_FLOAT_64,
	        PT_FLOAT_MAX,
	        PT_COMPLEX_128,
	    };

		/** cast data type to EnumContainerType and EnumPrimitiveType */
		template<typename T>
		struct Type2Enum
		{
			static constexpr EnumContainerType e_containertype = CT_UNDEFINED;
			static constexpr EnumPrimitiveType e_primitivetype = PT_UNDEFINED;
		};

		template<>
		struct Type2Enum<int32_t>
		{
			static constexpr EnumContainerType e_containertype = CT_PRIMITIVE;
			static constexpr EnumPrimitiveType e_primitivetype = PT_INT_32;
		};

		template<>
		struct Type2Enum<float64_t>
		{
			static constexpr EnumContainerType e_containertype = CT_PRIMITIVE;
			static constexpr EnumPrimitiveType e_primitivetype = PT_FLOAT_64;
		};

		template<>
		struct Type2Enum<SGVector<int32_t> >
		{
			static constexpr EnumContainerType e_containertype = CT_SGVECTOR;
			static constexpr EnumPrimitiveType e_primitivetype = PT_INT_32;
		};

		template<>
		struct Type2Enum<SGVector<float64_t> >
		{
			static constexpr EnumContainerType e_containertype = CT_SGVECTOR;
			static constexpr EnumPrimitiveType e_primitivetype = PT_FLOAT_64;
		};

		/** @brief data structure that saves the EnumContainerType
		 * and EnumPrimitiveType information of Any object
		 */
		struct DataType
		{
			EnumContainerType e_containertype;
			EnumPrimitiveType e_primitivetype;

			template<typename T>
			void set()
			{
				e_containertype = Type2Enum<T>::e_containertype;
				e_primitivetype = Type2Enum<T>::e_primitivetype;
			}
		};
	}
#endif //SWIG

	/** Converts compiler-dependent name of class to
	 * something human readable.
	 * @return human readable name of class
	 */
	template <typename T>
	std::string demangledType()
	{
		size_t length;
		int status;
		char* demangled = abi::__cxa_demangle(typeid(T).name(), nullptr, &length, &status);
		std::string demangled_string(demangled);
		free(demangled);
		return demangled_string;
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

		/** Clears storage.
		 * @param storage pointer to a pointer to storage
		 */
		virtual void clear(void** storage) const = 0;

		/** Returns type-name as string.
		 * @return name of type class
		 */
		virtual std::string type() const = 0;

		/** Compares type.
		 * @param ti type information
		 * @return true if type matches
		 */
		virtual bool matches(const std::type_info& ti) const = 0;

		/** Compares two storages.
		 * @param storage pointer to a pointer to storage
		 * @param other_storage pointer to a pointer to another storage
		 * @return true if both storages have same value
		 */
		virtual bool equals(void** storage, void** other_storage) const = 0;
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
			*(storage) = new T(*reinterpret_cast<T const*>(v));
		}

		/** Clears storage.
		 * @param storage pointer to a pointer to storage
		 */
		virtual void clear(void** storage) const
		{
			delete reinterpret_cast<T*>(*storage);
		}

		/** Returns type-name as string.
		 * @return name of type class
		 */
		virtual std::string type() const
		{
			return demangledType<T>();
		}

		/** Compares type.
		 * @param ti type information
		 * @return true if type matches
		 */
		virtual bool matches(const std::type_info& ti) const
		{
			return typeid(T) == ti;
		}

		/** Compares two storages.
		 * @param storage pointer to a pointer to storage
		 * @param other_storage pointer to a pointer to another storage
		 * @return true if both storages have same value
		 */
		bool equals(void** storage, void** other_storage) const
		{
			int typed_storage = *(reinterpret_cast<int*>(*storage));
			int typed_other_storage = *(reinterpret_cast<int*>(*other_storage));
			return typed_storage == typed_other_storage;
		}
	};

	/** @brief Allows to store objects of arbitrary types
	 * by using a BaseAnyPolicy and provides a type agnostic API.
	 * See its usage in CSGObject::Self, CSGObject::set(), CSGObject::get()
	 * and CSGObject::has().
	 * .
	 */
	class Any
	{
	public:
		/** Used to denote an empty Any object */
		struct Empty;

		/** Constructor */
		Any() : policy(select_policy<Empty>()), storage(nullptr)
		{
		}

		/** Constructor to copy value */
		template <typename T>
		explicit Any(const T& v) : policy(select_policy<T>()), storage(nullptr)
		{
			policy->set(&storage, &v);
			m_datatype.set<T>();
		}

		/** Copy constructor */
		Any(const Any& other) : policy(other.policy), storage(nullptr)
		{
			policy->set(&storage, other.storage);
			m_datatype = other.m_datatype;
		}

		/** Assignment operator
		 * @param other another Any object
		 * @return Any object
		 */
		Any& operator=(const Any& other)
		{
			policy->clear(&storage);
			policy = other.policy;
			policy->set(&storage, other.storage);
			m_datatype = other.m_datatype;
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

#ifndef SWIG // SWIG should skip this part
		/** Cast storage data to selected type and save the data to Archive
		 *
		 * @param ar Archive type
		 */
		template <class Archive, class Type>
		void cereal_save_helper(Archive& ar) const
		{
			ar(*(reinterpret_cast<Type*>(storage)));
		}

		template <class Archive>
		void cereal_complex_save_helper(Archive& ar) const
		{
			float64_t* temp = reinterpret_cast<float64_t*>(storage);
			ar(temp[0]);
			ar(temp[1]);
		}

		/** save data with cereal save method
		 *
		 * @param ar Archive type
		 */
		template<class Archive>
		void cereal_save(Archive& ar) const
		{
			ar (m_datatype.e_containertype);
			ar (m_datatype.e_primitivetype);
			switch (m_datatype.e_containertype) {
			case serial::EnumContainerType::CT_PRIMITIVE:
				switch (m_datatype.e_primitivetype) {
				case serial::EnumPrimitiveType::PT_BOOL_TYPE:
					cereal_save_helper<Archive, bool>(ar);
					break;
				case serial::EnumPrimitiveType::PT_CHAR_TYPE:
					cereal_save_helper<Archive, char>(ar);
					break;
				case serial::EnumPrimitiveType::PT_INT_8:
					cereal_save_helper<Archive, int8_t>(ar);
					break;
				case serial::EnumPrimitiveType::PT_UINT_8:
					cereal_save_helper<Archive, uint8_t>(ar);
					break;
				case serial::EnumPrimitiveType::PT_INT_16:
					cereal_save_helper<Archive, int16_t>(ar);
					break;
				case serial::EnumPrimitiveType::PT_UINT_16:
					cereal_save_helper<Archive, int32_t>(ar);
					break;
				case serial::EnumPrimitiveType::PT_INT_32:
					cereal_save_helper<Archive, int32_t>(ar);
					break;
				case serial::EnumPrimitiveType::PT_UINT_32:
					cereal_save_helper<Archive, uint32_t>(ar);
					break;
				case serial::EnumPrimitiveType::PT_INT_64:
					cereal_save_helper<Archive, int64_t>(ar);
					break;
				case serial::EnumPrimitiveType::PT_UINT_64:
					cereal_save_helper<Archive, uint64_t>(ar);
					break;
				case serial::EnumPrimitiveType::PT_FLOAT_32:
					cereal_save_helper<Archive, float32_t>(ar);
					break;
				case serial::EnumPrimitiveType::PT_FLOAT_64:
					cereal_save_helper<Archive, float64_t>(ar);
					break;
				case serial::EnumPrimitiveType::PT_FLOAT_MAX:
					cereal_save_helper<Archive, floatmax_t>(ar);
					break;
				case serial::EnumPrimitiveType::PT_COMPLEX_128:
					cereal_complex_save_helper<Archive>(ar);
					break;
				case serial::EnumPrimitiveType::PT_UNDEFINED:
					SG_SERROR("Type error: undefined data type cannot be serialized.\n");
					break;
				}
				break;

			case serial::EnumContainerType::CT_SGVECTOR:
				switch (m_datatype.e_primitivetype) {
				case serial::EnumPrimitiveType::PT_BOOL_TYPE:
					cereal_save_helper<Archive, SGVector<bool> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_CHAR_TYPE:
					cereal_save_helper<Archive, SGVector<char> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_INT_8:
					cereal_save_helper<Archive, SGVector<int8_t> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_UINT_8:
					cereal_save_helper<Archive, SGVector<uint8_t> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_INT_16:
					cereal_save_helper<Archive, SGVector<int16_t> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_UINT_16:
					cereal_save_helper<Archive, SGVector<int32_t> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_INT_32:
					cereal_save_helper<Archive, SGVector<int32_t> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_UINT_32:
					cereal_save_helper<Archive, SGVector<uint32_t> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_INT_64:
					cereal_save_helper<Archive, SGVector<int64_t> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_UINT_64:
					cereal_save_helper<Archive, SGVector<uint64_t> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_FLOAT_32:
					cereal_save_helper<Archive, SGVector<float32_t> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_FLOAT_64:
					cereal_save_helper<Archive, SGVector<float64_t> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_FLOAT_MAX:
					cereal_save_helper<Archive, SGVector<floatmax_t> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_COMPLEX_128:
					cereal_save_helper<Archive, SGVector<complex128_t> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_UNDEFINED:
					SG_SERROR("Type error: undefined data type cannot be serialized.\n");
					break;
				}
				break;

			case serial::EnumContainerType::CT_SGMATRIX:
				switch (m_datatype.e_primitivetype) {
				case serial::EnumPrimitiveType::PT_BOOL_TYPE:
					cereal_save_helper<Archive, SGMatrix<bool> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_CHAR_TYPE:
					cereal_save_helper<Archive, SGMatrix<char> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_INT_8:
					cereal_save_helper<Archive, SGMatrix<int8_t> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_UINT_8:
					cereal_save_helper<Archive, SGMatrix<uint8_t> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_INT_16:
					cereal_save_helper<Archive, SGMatrix<int16_t> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_UINT_16:
					cereal_save_helper<Archive, SGMatrix<int32_t> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_INT_32:
					cereal_save_helper<Archive, SGMatrix<int32_t> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_UINT_32:
					cereal_save_helper<Archive, SGMatrix<uint32_t> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_INT_64:
					cereal_save_helper<Archive, SGMatrix<int64_t> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_UINT_64:
					cereal_save_helper<Archive, SGMatrix<uint64_t> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_FLOAT_32:
					cereal_save_helper<Archive, SGMatrix<float32_t> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_FLOAT_64:
					cereal_save_helper<Archive, SGMatrix<float64_t> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_FLOAT_MAX:
					cereal_save_helper<Archive, SGMatrix<floatmax_t> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_COMPLEX_128:
					cereal_save_helper<Archive, SGMatrix<complex128_t> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_UNDEFINED:
					SG_SERROR("Type error: undefined data type cannot be serialized.\n");
					break;
				}
				break;
			case serial::EnumContainerType::CT_UNDEFINED:
				SG_SERROR("Type error: undefined container type cannot be serialized.\n");
				break;
			}
		}

		/** Load data from archive and cast to Any type
		 *
		 * @param ar Archive type
		 */
		template <class Archive, class Type>
		void cereal_load_helper(Archive& ar)
		{
			Type temp;
			ar(temp);
			policy->clear(&storage);
			policy->set(&storage, &temp);
		}

		template <class Archive>
		void cereal_complex_load_helper(Archive& ar)
		{
			complex128_t data_temp;
			float64_t* temp = reinterpret_cast<float64_t*>(&data_temp);
			ar(temp[0]);
			ar(temp[1]);
			policy->clear(&storage);
			policy->set(&storage, &data_temp);
		}

		/** load data from archive with cereal load method
		 *
		 * @param ar Archive type
		 */
		template<class Archive>
		void cereal_load(Archive& ar)
		{
			ar(m_datatype.e_containertype);
			ar(m_datatype.e_primitivetype);
			switch (m_datatype.e_containertype) {
			case serial::EnumContainerType::CT_PRIMITIVE:
				switch (m_datatype.e_primitivetype) {
				case serial::EnumPrimitiveType::PT_BOOL_TYPE:
					cereal_load_helper<Archive, bool>(ar);
					break;
				case serial::EnumPrimitiveType::PT_CHAR_TYPE:
					cereal_load_helper<Archive, char>(ar);
					break;
				case serial::EnumPrimitiveType::PT_INT_8:
					cereal_load_helper<Archive, int8_t>(ar);
					break;
				case serial::EnumPrimitiveType::PT_UINT_8:
					cereal_load_helper<Archive, uint8_t>(ar);
					break;
				case serial::EnumPrimitiveType::PT_INT_16:
					cereal_load_helper<Archive, int16_t>(ar);
					break;
				case serial::EnumPrimitiveType::PT_UINT_16:
					cereal_load_helper<Archive, int32_t>(ar);
					break;
				case serial::EnumPrimitiveType::PT_INT_32:
					cereal_load_helper<Archive, int32_t>(ar);
					break;
				case serial::EnumPrimitiveType::PT_UINT_32:
					cereal_load_helper<Archive, uint32_t>(ar);
					break;
				case serial::EnumPrimitiveType::PT_INT_64:
					cereal_load_helper<Archive, int64_t>(ar);
					break;
				case serial::EnumPrimitiveType::PT_UINT_64:
					cereal_load_helper<Archive, uint64_t>(ar);
					break;
				case serial::EnumPrimitiveType::PT_FLOAT_32:
					cereal_load_helper<Archive, float32_t>(ar);
					break;
				case serial::EnumPrimitiveType::PT_FLOAT_64:
					cereal_load_helper<Archive, float64_t>(ar);
					break;
				case serial::EnumPrimitiveType::PT_FLOAT_MAX:
					cereal_load_helper<Archive, floatmax_t>(ar);
					break;
				case serial::EnumPrimitiveType::PT_COMPLEX_128:
					cereal_complex_load_helper<Archive>(ar);
					break;
				case serial::EnumPrimitiveType::PT_UNDEFINED:
					SG_SERROR("Error: undefined container type cannot be loaded.\n");
					break;
				}
				break;

			case serial::EnumContainerType::CT_SGVECTOR:
				switch (m_datatype.e_primitivetype) {
				case serial::EnumPrimitiveType::PT_BOOL_TYPE:
					cereal_load_helper<Archive, SGVector<bool> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_CHAR_TYPE:
					cereal_load_helper<Archive, SGVector<char> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_INT_8:
					cereal_load_helper<Archive, SGVector<int8_t> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_UINT_8:
					cereal_load_helper<Archive, SGVector<uint8_t> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_INT_16:
					cereal_load_helper<Archive, SGVector<int16_t> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_UINT_16:
					cereal_load_helper<Archive, SGVector<int32_t> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_INT_32:
					cereal_load_helper<Archive, SGVector<int32_t> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_UINT_32:
					cereal_load_helper<Archive, SGVector<uint32_t> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_INT_64:
					cereal_load_helper<Archive, SGVector<int64_t> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_UINT_64:
					cereal_load_helper<Archive, SGVector<uint64_t> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_FLOAT_32:
					cereal_load_helper<Archive, SGVector<float32_t> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_FLOAT_64:
					cereal_load_helper<Archive, SGVector<float64_t> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_FLOAT_MAX:
					cereal_load_helper<Archive, SGVector<floatmax_t> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_COMPLEX_128:
					cereal_load_helper<Archive, SGVector<complex128_t> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_UNDEFINED:
					SG_SERROR("Error: undefined container type cannot be loaded.\n");
					break;
				}
				break;

			case serial::EnumContainerType::CT_SGMATRIX:
				switch (m_datatype.e_primitivetype) {
				case serial::EnumPrimitiveType::PT_BOOL_TYPE:
					cereal_load_helper<Archive, SGMatrix<bool> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_CHAR_TYPE:
					cereal_load_helper<Archive, SGMatrix<char> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_INT_8:
					cereal_load_helper<Archive, SGMatrix<int8_t> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_UINT_8:
					cereal_load_helper<Archive, SGMatrix<uint8_t> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_INT_16:
					cereal_load_helper<Archive, SGMatrix<int16_t> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_UINT_16:
					cereal_load_helper<Archive, SGMatrix<int32_t> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_INT_32:
					cereal_load_helper<Archive, SGMatrix<int32_t> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_UINT_32:
					cereal_load_helper<Archive, SGMatrix<uint32_t> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_INT_64:
					cereal_load_helper<Archive, SGMatrix<int64_t> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_UINT_64:
					cereal_load_helper<Archive, SGMatrix<uint64_t> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_FLOAT_32:
					cereal_load_helper<Archive, SGMatrix<float32_t> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_FLOAT_64:
					cereal_load_helper<Archive, SGMatrix<float64_t> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_FLOAT_MAX:
					cereal_load_helper<Archive, SGMatrix<floatmax_t> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_COMPLEX_128:
					cereal_load_helper<Archive, SGMatrix<complex128_t> >(ar);
					break;
				case serial::EnumPrimitiveType::PT_UNDEFINED:
					SG_SERROR("Error: undefined container type cannot be loaded.\n");
					break;
				}
				break;

			default:
				SG_SERROR("Error: undefined container type cannot be loaded.\n");
				break;
			}
		}
#endif //SWIG

		/** Casts hidden value to provided type, fails otherwise.
		 * @return type-casted value
		 */
		template <typename T>
		T& as() const
		{
			if (same_type<T>())
				return *(reinterpret_cast<T*>(storage));
			else
				throw std::logic_error("Bad cast to " + demangledType<T>() +
					" but the type is " + policy->type());
		}

		/** @return true if type is same. */
		template <typename T>
		inline bool same_type() const
		{
			return (policy == select_policy<T>()) || same_type_fallback<T>();
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

	private:
		template <typename T>
		static BaseAnyPolicy* select_policy()
		{
			typedef PointerValueAnyPolicy<T> Policy;
			static Policy policy;
			return &policy;
		}

		BaseAnyPolicy* policy;
		void* storage;

#ifndef SWIG // SWIG should skip this part
		/** Enum structure that saves the type information of Any */
		serial::DataType m_datatype;
#endif //#ifndef SWIG // SWIG should skip this part
	};

	inline bool operator==(const Any& lhs, const Any& rhs)
	{
		void* lhs_storage = lhs.storage;
		void* rhs_storage = rhs.storage;
		return lhs.policy == rhs.policy and
		    lhs.policy->equals(&lhs_storage, &rhs_storage);
	}

	inline bool operator!=(const Any& lhs, const Any& rhs)
	{
		return !(lhs == rhs);
	}

	/** Used to denote an empty Any object */
	struct Any::Empty
	{
		/** Equality operator */
		bool operator==(const Empty& other) const
		{
			return true;
		}
	};

	/** Erases value type i.e. converts it to Any
	 * For input object of any type, it returns an Any object
	 * which stores the input object's raw value. It saves the type
	 * information internally to be recalled later by using recall_type().
	 *
	 * @param v value
	 * @return Any object with the input value
	 */
	template <typename T>
	inline Any erase_type(const T& v)
	{
		return Any(v);
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
	inline T recall_type(const Any& any)
	{
		return any.as<T>();
	}

}

#endif  //_ANY_H_
