/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_TYPE_CASE_H
#define SHOGUN_TYPE_CASE_H

#include <typeindex>
#include <unordered_map>

#include <shogun/lib/SGVector.h>
#include <shogun/lib/any.h>
#include <shogun/lib/type_list.h>

using namespace shogun;

namespace shogun
{

	typedef Types<
	    bool, char, int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t,
	    int64_t, uint64_t, float32_t, float64_t, floatmax_t>::type
	    SG_SCALAR_TYPES;
	typedef Types<SGVector<float32_t>>::type SG_VECTOR_TYPES;

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
		PT_SGVECTOR_FLOAT32 = 16,
		PT_UNDEFINED = 17
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
SG_PRIMITIVE_TYPE(SGVector<float32_t>, TYPE::PT_SGVECTOR_FLOAT32)

#undef SG_PRIMITIVE_TYPE

namespace shogun
{
	typedef std::unordered_map<std::type_index, TYPE> typemap;

	namespace type_internal
	{
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

		TYPE get_type(const Any& any, const typemap& map)
		{
			auto type = std::type_index(any.type_info());
			typemap::const_iterator it = map.find(type);

			return it == map.end() ? TYPE::PT_UNDEFINED : map.at(type);
		}

		template <typename TypeList, typename Lambda>
		typename std::enable_if<
		    std::is_same<TypeList, Types0>::value, void>::type
		sg_type_finder(const Any& any, TYPE type, Lambda func)
		{
			SG_SERROR(
			    "Unsupported type %s",
			    demangled_type(any.type_info().name()).c_str())
		}

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
            ADD_TYPE_TO_MAP(SGVector<float32_t>, TYPE::PT_SGVECTOR_FLOAT32)
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

    /** Checks if the underlying type of an Any instance is a scalar and
     * executes the given lambda. The type is checked
     * against a typemap and if it isn't found there
     * a ShogunException is thrown.
     * The lambda function must have a specific signature,
     * where it has one auto deduced argument. This argument will have
     * the same underlying type of the Any instance and can then
     * be used inside the lambda:
     * @code
     * auto f = [&any](auto type) {
     *     std::cout << any_cast<decltype(type)>(any);
     * };
     * @endcode
     *
     * @see sg_for_each_vector_type
	 *
     * @param any Any object
	 * @param typesmap check the underlying type of Any given this map
     * @param func lambda to execute if underlying type is found
	 */
	template <typename Lambda>
	void sg_for_each_scalar_type(const Any& any, const typemap& typesmap, Lambda func)
	{
		TYPE type = type_internal::get_type(any, typesmap);
		if (type == TYPE::PT_UNDEFINED)
			SG_SERROR(
			    "Type %s is not part of %s",
			    demangled_type(any.type_info().name()).c_str(),
			    type_internal::print_map(typesmap).c_str())
		else
			type_internal::sg_type_finder<SG_SCALAR_TYPES>(any, type, func);
	}

    /** Checks if the underlying type of an Any instance is a vector and
     * executes the given lambda.
     *
     * @see sg_for_each_scalar_type
     *
     * @param any Any object
     * @param typesmap check the underlying type of Any given this map
     * @param func lambda to execute if underlying type is found
     */
	template <typename Lambda>
	void sg_for_each_vector_type(const Any& any, const typemap& typesmap, Lambda func)
	{
		TYPE type = type_internal::get_type(any, typesmap);
		if (type == TYPE::PT_UNDEFINED)
			SG_SERROR(
			    "Type %s is not part of %s",
			    demangled_type(any.type_info().name()).c_str(),
			    type_internal::print_map(typesmap).c_str())
		else
			type_internal::sg_type_finder<SG_VECTOR_TYPES>(any, type, func);
	}

} // namespace shogun

#endif // SHOGUN_TYPE_CASE_H
