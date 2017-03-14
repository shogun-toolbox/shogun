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

#define CASE_PRIMITIVE_SAVE(datatype) {\
	case (serial::PT_##datatype): \
		cereal_save_helper<Archive, datatype>(ar); \
		break; \
} \

#define CASE_SGOBJECT_SAVE(objecttype, datatype) {\
	case (serial::PT_##datatype): \
		cereal_save_helper<Archive, objecttype<datatype> >(ar); \
		break; \
} \

#define CASE_PRIMITIVE_LOAD(datatype) {\
	case (serial::PT_##datatype): \
		cereal_load_helper<Archive, datatype>(ar); \
		break; \
} \

#define CASE_SGOBJECT_LOAD(objecttype, datatype) {\
	case (serial::PT_##datatype): \
		cereal_load_helper<Archive, objecttype<datatype> >(ar); \
		break; \
} \

namespace serial
{
		enum EnumContainerType
		{
			CT_UNDEFINED,
			CT_PRIMITIVE,
			CT_SGVector,
			CT_SGMatrix
		};

		enum EnumPrimitiveType
		{
			PT_UNDEFINED,
			PT_bool,
			PT_char,
			PT_int8_t,
			PT_uint8_t,
			PT_int16_t,
			PT_uint16_t,
			PT_int32_t,
			PT_uint32_t,
			PT_int64_t,
			PT_uint64_t,
			PT_float32_t,
			PT_float64_t,
			PT_floatmax_t,
			PT_complex128_t,
		};

		/** cast data type to EnumContainerType and EnumPrimitiveType */
		template<typename T>
		struct Type2Enum
		{
			static constexpr EnumContainerType e_containertype = CT_UNDEFINED;
			static constexpr EnumPrimitiveType e_primitivetype = PT_UNDEFINED;
		};

		template<>
		struct Type2Enum<bool>
		{
			static constexpr EnumContainerType e_containertype = CT_PRIMITIVE;
			static constexpr EnumPrimitiveType e_primitivetype = PT_bool;
		};

		template<>
		struct Type2Enum<char>
		{
			static constexpr EnumContainerType e_containertype = CT_PRIMITIVE;
			static constexpr EnumPrimitiveType e_primitivetype = PT_char;
		};

		template<>
		struct Type2Enum<int8_t>
		{
			static constexpr EnumContainerType e_containertype = CT_PRIMITIVE;
			static constexpr EnumPrimitiveType e_primitivetype = PT_int8_t;
		};

		template<>
		struct Type2Enum<uint8_t>
		{
			static constexpr EnumContainerType e_containertype = CT_PRIMITIVE;
			static constexpr EnumPrimitiveType e_primitivetype = PT_uint8_t;
		};

		template<>
		struct Type2Enum<int16_t>
		{
			static constexpr EnumContainerType e_containertype = CT_PRIMITIVE;
			static constexpr EnumPrimitiveType e_primitivetype = PT_int16_t;
		};

		template<>
		struct Type2Enum<uint16_t>
		{
			static constexpr EnumContainerType e_containertype = CT_PRIMITIVE;
			static constexpr EnumPrimitiveType e_primitivetype = PT_uint16_t;
		};

		template<>
		struct Type2Enum<int32_t>
		{
			static constexpr EnumContainerType e_containertype = CT_PRIMITIVE;
			static constexpr EnumPrimitiveType e_primitivetype = PT_int32_t;
		};

		template<>
		struct Type2Enum<uint32_t>
		{
			static constexpr EnumContainerType e_containertype = CT_PRIMITIVE;
			static constexpr EnumPrimitiveType e_primitivetype = PT_uint32_t;
		};

		template<>
		struct Type2Enum<int64_t>
		{
			static constexpr EnumContainerType e_containertype = CT_PRIMITIVE;
			static constexpr EnumPrimitiveType e_primitivetype = PT_int64_t;
		};

		template<>
		struct Type2Enum<uint64_t>
		{
			static constexpr EnumContainerType e_containertype = CT_PRIMITIVE;
			static constexpr EnumPrimitiveType e_primitivetype = PT_uint64_t;
		};

		template<>
		struct Type2Enum<float32_t>
		{
			static constexpr EnumContainerType e_containertype = CT_PRIMITIVE;
			static constexpr EnumPrimitiveType e_primitivetype = PT_float32_t;
		};

		template<>
		struct Type2Enum<float64_t>
		{
			static constexpr EnumContainerType e_containertype = CT_PRIMITIVE;
			static constexpr EnumPrimitiveType e_primitivetype = PT_float64_t;
		};

		template<>
		struct Type2Enum<floatmax_t>
		{
			static constexpr EnumContainerType e_containertype = CT_PRIMITIVE;
			static constexpr EnumPrimitiveType e_primitivetype = PT_floatmax_t;
		};

		template<>
		struct Type2Enum<complex128_t>
		{
			static constexpr EnumContainerType e_containertype = CT_PRIMITIVE;
			static constexpr EnumPrimitiveType e_primitivetype = PT_complex128_t;
		};

		template<>
		struct Type2Enum<SGVector<bool> >
		{
			static constexpr EnumContainerType e_containertype = CT_SGVector;
			static constexpr EnumPrimitiveType e_primitivetype = PT_bool;
		};

		template<>
		struct Type2Enum<SGVector<char> >
		{
			static constexpr EnumContainerType e_containertype = CT_SGVector;
			static constexpr EnumPrimitiveType e_primitivetype = PT_char;
		};

		template<>
		struct Type2Enum<SGVector<int8_t> >
		{
			static constexpr EnumContainerType e_containertype = CT_SGVector;
			static constexpr EnumPrimitiveType e_primitivetype = PT_int8_t;
		};

		template<>
		struct Type2Enum<SGVector<uint8_t> >
		{
			static constexpr EnumContainerType e_containertype = CT_SGVector;
			static constexpr EnumPrimitiveType e_primitivetype = PT_uint8_t;
		};

		template<>
		struct Type2Enum<SGVector<int16_t> >
		{
			static constexpr EnumContainerType e_containertype = CT_SGVector;
			static constexpr EnumPrimitiveType e_primitivetype = PT_int16_t;
		};

		template<>
		struct Type2Enum<SGVector<uint16_t> >
		{
			static constexpr EnumContainerType e_containertype = CT_SGVector;
			static constexpr EnumPrimitiveType e_primitivetype = PT_uint16_t;
		};

		template<>
		struct Type2Enum<SGVector<int32_t> >
		{
			static constexpr EnumContainerType e_containertype = CT_SGVector;
			static constexpr EnumPrimitiveType e_primitivetype = PT_int32_t;
		};

		template<>
		struct Type2Enum<SGVector<uint32_t> >
		{
			static constexpr EnumContainerType e_containertype = CT_SGVector;
			static constexpr EnumPrimitiveType e_primitivetype = PT_uint32_t;
		};

		template<>
		struct Type2Enum<SGVector<int64_t> >
		{
			static constexpr EnumContainerType e_containertype = CT_SGVector;
			static constexpr EnumPrimitiveType e_primitivetype = PT_int64_t;
		};

		template<>
		struct Type2Enum<SGVector<uint64_t> >
		{
			static constexpr EnumContainerType e_containertype = CT_SGVector;
			static constexpr EnumPrimitiveType e_primitivetype = PT_uint64_t;
		};

		template<>
		struct Type2Enum<SGVector<float32_t> >
		{
			static constexpr EnumContainerType e_containertype = CT_SGVector;
			static constexpr EnumPrimitiveType e_primitivetype = PT_float32_t;
		};

		template<>
		struct Type2Enum<SGVector<float64_t> >
		{
			static constexpr EnumContainerType e_containertype = CT_SGVector;
			static constexpr EnumPrimitiveType e_primitivetype = PT_float64_t;
		};

		template<>
		struct Type2Enum<SGVector<floatmax_t> >
		{
			static constexpr EnumContainerType e_containertype = CT_SGVector;
			static constexpr EnumPrimitiveType e_primitivetype = PT_floatmax_t;
		};

		template<>
		struct Type2Enum<SGVector<complex128_t> >
		{
			static constexpr EnumContainerType e_containertype = CT_SGVector;
			static constexpr EnumPrimitiveType e_primitivetype = PT_complex128_t;
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
				CASE_PRIMITIVE_SAVE(bool);
				CASE_PRIMITIVE_SAVE(char);
				CASE_PRIMITIVE_SAVE(int8_t);
				CASE_PRIMITIVE_SAVE(uint8_t);
				CASE_PRIMITIVE_SAVE(int16_t);
				CASE_PRIMITIVE_SAVE(uint16_t);
				CASE_PRIMITIVE_SAVE(int32_t);
				CASE_PRIMITIVE_SAVE(uint32_t);
				CASE_PRIMITIVE_SAVE(int64_t);
				CASE_PRIMITIVE_SAVE(uint64_t);
				CASE_PRIMITIVE_SAVE(float32_t);
				CASE_PRIMITIVE_SAVE(float64_t);
				CASE_PRIMITIVE_SAVE(floatmax_t);
				case serial::EnumPrimitiveType::PT_complex128_t:
					cereal_complex_save_helper<Archive>(ar);
					break;
				case serial::EnumPrimitiveType::PT_UNDEFINED:
					SG_SERROR("Type error: undefined data type cannot be serialized.\n");
					break;
				}
				break;
			case serial::EnumContainerType::CT_SGVector:
				switch (m_datatype.e_primitivetype) {
				CASE_SGOBJECT_SAVE(SGVector, bool);
				CASE_SGOBJECT_SAVE(SGVector, char);
				CASE_SGOBJECT_SAVE(SGVector, int8_t);
				CASE_SGOBJECT_SAVE(SGVector, uint8_t);
				CASE_SGOBJECT_SAVE(SGVector, int16_t);
				CASE_SGOBJECT_SAVE(SGVector, uint16_t);
				CASE_SGOBJECT_SAVE(SGVector, int32_t);
				CASE_SGOBJECT_SAVE(SGVector, uint32_t);
				CASE_SGOBJECT_SAVE(SGVector, int64_t);
				CASE_SGOBJECT_SAVE(SGVector, uint64_t);
				CASE_SGOBJECT_SAVE(SGVector, float32_t);
				CASE_SGOBJECT_SAVE(SGVector, float64_t);
				CASE_SGOBJECT_SAVE(SGVector, floatmax_t);
				CASE_SGOBJECT_SAVE(SGVector, complex128_t);
				case serial::EnumPrimitiveType::PT_UNDEFINED:
					SG_SERROR("Type error: undefined data type cannot be serialized.\n");
					break;
				}
				break;
			case serial::EnumContainerType::CT_SGMatrix:
				SG_SWARNING("SGMatrix serializatino method not implemented.\n");
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
				CASE_PRIMITIVE_LOAD(bool);
				CASE_PRIMITIVE_LOAD(char);
				CASE_PRIMITIVE_LOAD(int8_t);
				CASE_PRIMITIVE_LOAD(uint8_t);
				CASE_PRIMITIVE_LOAD(int16_t);
				CASE_PRIMITIVE_LOAD(uint16_t);
				CASE_PRIMITIVE_LOAD(int32_t);
				CASE_PRIMITIVE_LOAD(uint32_t);
				CASE_PRIMITIVE_LOAD(int64_t);
				CASE_PRIMITIVE_LOAD(uint64_t);
				CASE_PRIMITIVE_LOAD(float32_t);
				CASE_PRIMITIVE_LOAD(float64_t);
				CASE_PRIMITIVE_LOAD(floatmax_t);
				case serial::EnumPrimitiveType::PT_complex128_t:
					cereal_complex_load_helper<Archive>(ar);
					break;
				default:
					SG_SERROR("Error: undefined data type cannot be loaded.\n");
					break;
				}
				break;

			case serial::EnumContainerType::CT_SGVector:
				switch (m_datatype.e_primitivetype) {
				CASE_SGOBJECT_LOAD(SGVector, bool);
				CASE_SGOBJECT_LOAD(SGVector, char);
				CASE_SGOBJECT_LOAD(SGVector, int8_t);
				CASE_SGOBJECT_LOAD(SGVector, uint8_t);
				CASE_SGOBJECT_LOAD(SGVector, int16_t);
				CASE_SGOBJECT_LOAD(SGVector, uint16_t);
				CASE_SGOBJECT_LOAD(SGVector, int32_t);
				CASE_SGOBJECT_LOAD(SGVector, uint32_t);
				CASE_SGOBJECT_LOAD(SGVector, int64_t);
				CASE_SGOBJECT_LOAD(SGVector, uint64_t);
				CASE_SGOBJECT_LOAD(SGVector, float32_t);
				CASE_SGOBJECT_LOAD(SGVector, float64_t);
				CASE_SGOBJECT_LOAD(SGVector, floatmax_t);
				CASE_SGOBJECT_LOAD(SGVector, complex128_t);
				case serial::EnumPrimitiveType::PT_UNDEFINED:
					SG_SERROR("Type error: undefined data type cannot be serialized.\n");
					break;
				}
				break;

			case serial::EnumContainerType::CT_SGMatrix:
				SG_SWARNING("SGMatrix serializatino method not implemented.\n")

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
