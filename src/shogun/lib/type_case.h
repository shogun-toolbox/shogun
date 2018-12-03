/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_TYPE_CASE_H
#define SHOGUN_TYPE_CASE_H

#include <typeindex>
#include <unordered_map>

#include <shogun/lib/any.h>
#include <shogun/lib/type_list.h>

using namespace shogun;

namespace shogun
{
	typedef Types<
	    bool, char, int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t,
	    int64_t, uint64_t, float32_t, float64_t, floatmax_t>::type SG_TYPES;

	enum class TYPE
	{
		PT_BOOL = 1,
		PT_CHAR = 2,
		PT_INT8 = 3,
		PT_UINT8 = 4,
		PT_INT16 = 5,
		PT_UINT16 = 6,
		PT_INT32 = 7,
		PT_UINT32 = 8,
		PT_INT64 = 9,
		PT_UINT64 = 10,
		PT_FLOAT32 = 11,
		PT_FLOAT64 = 12,
		PT_FLOATMAX = 13,
		PT_SGOBJECT = 14,
		PT_COMPLEX128 = 15,
		PT_UNDEFINED = 16
	};
} // namespace shogun

template <typename T>
struct sg_primitive_type
{
};

#define SG_PRIMITIVE_TYPE(T, ptype)                                            \
	template <>                                                                \
	struct sg_primitive_type<T>                                                \
	{                                                                          \
		static constexpr TYPE value = ptype;                                   \
	};

SG_PRIMITIVE_TYPE(bool, TYPE::PT_BOOL)
SG_PRIMITIVE_TYPE(char, TYPE::PT_CHAR)
SG_PRIMITIVE_TYPE(int8_t, TYPE::PT_INT8)
SG_PRIMITIVE_TYPE(uint8_t, TYPE::PT_UINT8)
SG_PRIMITIVE_TYPE(int16_t, TYPE::PT_INT16)
SG_PRIMITIVE_TYPE(uint16_t, TYPE::PT_UINT16)
SG_PRIMITIVE_TYPE(int32_t, TYPE::PT_INT32)
SG_PRIMITIVE_TYPE(uint32_t, TYPE::PT_UINT32)
SG_PRIMITIVE_TYPE(int64_t, TYPE::PT_INT64)
SG_PRIMITIVE_TYPE(uint64_t, TYPE::PT_UINT64)
SG_PRIMITIVE_TYPE(float32_t, TYPE::PT_FLOAT32)
SG_PRIMITIVE_TYPE(float64_t, TYPE::PT_FLOAT64)
SG_PRIMITIVE_TYPE(floatmax_t, TYPE::PT_FLOATMAX)
SG_PRIMITIVE_TYPE(complex128_t, TYPE::PT_COMPLEX128)

#undef SG_PRIMITIVE_TYPE

namespace shogun
{
	typedef std::unordered_map<std::type_index, TYPE> typemap;

	namespace type_internal
	{
		std::string demangled_type(const char* name)
		{
#ifdef HAVE_CXA_DEMANGLE
			size_t length;
			int status;
			char* demangled =
			    abi::__cxa_demangle(name, nullptr, &length, &status);
			std::string demangled_string(demangled);
			free(demangled);
#else
			std::string demangled_string(name);
#endif
			return demangled_string;
		}

		std::string print_map(const typemap& map)
		{
			auto msg = std::string("<");
			for (auto it : map)
			{
				msg += demangled_type(it.first.name());
				msg += ", ";
			}
			msg.resize(msg.size() - 2);
			msg += ">";
			return msg;
		}

		TYPE get_type(const Any& any, typemap& map)
		{
			auto type = std::type_index(any.type_info());
			typemap::const_iterator it = map.find(type);

			return it == map.end() ? TYPE::PT_UNDEFINED : map[type];
		}
	} // namespace type_internal

#define ADD_TYPE_TO_MAP(TYPENAME, TYPE_ENUM)                                   \
	{std::type_index(typeid(TYPENAME)), TYPE_ENUM},

	typemap sg_all_types = {
			ADD_TYPE_TO_MAP(bool, TYPE::PT_BOOL)
			ADD_TYPE_TO_MAP(char, TYPE::PT_CHAR)
			ADD_TYPE_TO_MAP(int8_t, TYPE::PT_INT8)
			ADD_TYPE_TO_MAP(uint8_t, TYPE::PT_UINT8)
			ADD_TYPE_TO_MAP(int16_t , TYPE::PT_INT16)
			ADD_TYPE_TO_MAP(uint16_t , TYPE::PT_UINT16)
			ADD_TYPE_TO_MAP(int32_t , TYPE::PT_INT32)
			ADD_TYPE_TO_MAP(uint32_t , TYPE::PT_UINT32)
			ADD_TYPE_TO_MAP(int64_t , TYPE::PT_INT64)
			ADD_TYPE_TO_MAP(uint64_t , TYPE::PT_UINT64)
			ADD_TYPE_TO_MAP(float32_t , TYPE::PT_FLOAT32)
			ADD_TYPE_TO_MAP(float64_t , TYPE::PT_FLOAT64)
			ADD_TYPE_TO_MAP(floatmax_t , TYPE::PT_FLOATMAX)
			ADD_TYPE_TO_MAP(complex128_t, TYPE::PT_COMPLEX128)
	};
	typemap sg_non_complex_types = {
			ADD_TYPE_TO_MAP(bool, TYPE::PT_BOOL)
			ADD_TYPE_TO_MAP(char, TYPE::PT_CHAR)
			ADD_TYPE_TO_MAP(int8_t, TYPE::PT_INT8)
			ADD_TYPE_TO_MAP(uint8_t, TYPE::PT_UINT8)
			ADD_TYPE_TO_MAP(int16_t , TYPE::PT_INT16)
			ADD_TYPE_TO_MAP(uint16_t , TYPE::PT_UINT16)
			ADD_TYPE_TO_MAP(int32_t , TYPE::PT_INT32)
			ADD_TYPE_TO_MAP(uint32_t , TYPE::PT_UINT32)
			ADD_TYPE_TO_MAP(int64_t , TYPE::PT_INT64)
			ADD_TYPE_TO_MAP(uint64_t , TYPE::PT_UINT64)
			ADD_TYPE_TO_MAP(float32_t , TYPE::PT_FLOAT32)
			ADD_TYPE_TO_MAP(float64_t , TYPE::PT_FLOAT64)
			ADD_TYPE_TO_MAP(floatmax_t , TYPE::PT_FLOATMAX)
	};
	typemap sg_real_types = {
			ADD_TYPE_TO_MAP(float32_t , TYPE::PT_FLOAT32)
			ADD_TYPE_TO_MAP(float64_t , TYPE::PT_FLOAT64)
			ADD_TYPE_TO_MAP(floatmax_t , TYPE::PT_FLOATMAX)
	};
	typemap sg_non_integer_types = {
			ADD_TYPE_TO_MAP(float32_t , TYPE::PT_FLOAT32)
			ADD_TYPE_TO_MAP(float64_t , TYPE::PT_FLOAT64)
			ADD_TYPE_TO_MAP(floatmax_t , TYPE::PT_FLOATMAX)
			ADD_TYPE_TO_MAP(complex128_t, TYPE::PT_COMPLEX128)
	};

#undef ADD_TYPE_TO_MAP
	template <typename TypeList, typename Lambda>
	typename std::enable_if<
	    (not std::is_same<TypeList, Types0>::value), void>::type
	sg_type_finder(const Any& any, TYPE type, Lambda func)
	{
		if (type == sg_primitive_type<typename TypeList::Head>::value)
		{
			typename TypeList::Head temporary_type_holder;
			func(temporary_type_holder);
		}
		else
		{
			sg_type_finder<typename TypeList::Tail>(any, type, func);
		}
	}

	template <typename TypeList, typename Lambda>
	typename std::enable_if<std::is_same<TypeList, Types0>::value, void>::type
	sg_type_finder(const Any& any, TYPE type, Lambda func)
	{
		SG_SERROR(
		    "Unsupported type %s",
		    type_internal::demangled_type(any.type_info().name()).c_str())
	}

	template <typename Lambda>
	void sg_for_each_type(const Any& any, typemap& typesmap, Lambda func)
	{
		TYPE type = type_internal::get_type(any, typesmap);
		if (type == TYPE::PT_UNDEFINED)
			SG_SERROR(
			    "Type %s is not part of %s",
			    type_internal::demangled_type(any.type_info().name()).c_str(),
			    type_internal::print_map(typesmap).c_str())
		else
			sg_type_finder<SG_TYPES>(any, type, func);
	}

} // namespace shogun

#endif // SHOGUN_TYPE_CASE_H
