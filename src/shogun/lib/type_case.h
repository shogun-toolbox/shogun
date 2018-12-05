/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_TYPE_CASE_H
#define SHOGUN_TYPE_CASE_H

#include <typeindex>
#include <unordered_map>

#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/any.h>
#include <shogun/lib/type_list.h>

using namespace shogun;

namespace shogun
{
	typedef Types<
		bool, char, int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t,
		int64_t, uint64_t, float32_t, float64_t, floatmax_t, SGVector<int32_t>,
		SGVector<int64_t>, SGVector<float32_t>, SGVector<float64_t>,
		SGVector<floatmax_t>, SGMatrix<int32_t>, SGMatrix<int64_t>,
		SGMatrix<float32_t>, SGMatrix<float64_t>, SGMatrix<floatmax_t>>::type
		SG_TYPES;

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
		PT_SGVECTOR_FLOAT64 = 17,
		PT_SGVECTOR_FLOATMAX = 18,
		PT_SGVECTOR_INT32 = 19,
		PT_SGVECTOR_INT64 = 20,
		PT_SGMATRIX_FLOAT32 = 21,
		PT_SGMATRIX_FLOAT64 = 22,
		PT_SGMATRIX_FLOATMAX = 23,
		PT_SGMATRIX_INT32 = 24,
		PT_SGMATRIX_INT64 = 25,
		PT_UNDEFINED = 26
	};

	enum class CONTAINER_TYPE
	{
		SCALAR = 1,
		VECTOR = 2,
		MATRIX = 3
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
		static constexpr TYPE pvalue = ptype;                                  \
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
SG_PRIMITIVE_TYPE(SGVector<float64_t>, TYPE::PT_SGVECTOR_FLOAT64)
SG_PRIMITIVE_TYPE(SGVector<floatmax_t>, TYPE::PT_SGVECTOR_FLOATMAX)
SG_PRIMITIVE_TYPE(SGVector<int32_t>, TYPE::PT_SGVECTOR_INT32)
SG_PRIMITIVE_TYPE(SGVector<int64_t>, TYPE::PT_SGVECTOR_INT64)
SG_PRIMITIVE_TYPE(SGMatrix<float32_t>, TYPE::PT_SGMATRIX_FLOAT32)
SG_PRIMITIVE_TYPE(SGMatrix<float64_t>, TYPE::PT_SGMATRIX_FLOAT64)
SG_PRIMITIVE_TYPE(SGMatrix<floatmax_t>, TYPE::PT_SGMATRIX_FLOATMAX)
SG_PRIMITIVE_TYPE(SGMatrix<int32_t>, TYPE::PT_SGMATRIX_INT32)
SG_PRIMITIVE_TYPE(SGMatrix<int64_t>, TYPE::PT_SGMATRIX_INT64)

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
			auto it = map.find(type);

			return it == map.end() ? TYPE::PT_UNDEFINED : map.at(type);
		}

		template <typename T>
		struct is_vector : public std::false_type
		{
		};
		template <typename T>
		struct is_vector<SGVector<T>> : public std::true_type
		{
		};

		template <typename T>
		struct is_matrix : public std::false_type
		{
		};
		template <typename T>
		struct is_matrix<SGMatrix<T>> : public std::true_type
		{
		};

		template <
			typename T, typename Lambda1, typename Lambda2, typename Lambda3>
		typename std::enable_if_t<
			std::is_scalar<T>::value and
				not std::is_same<Lambda1, std::nullptr_t>::value,
			void>
		execute_function(
			T value, Lambda1 scalar_func, Lambda2 vector_func,
			Lambda3 matrix_func)
		{
			scalar_func(value);
		}

		template <
			typename T, typename Lambda1, typename Lambda2, typename Lambda3>
		typename std::enable_if_t<
			is_vector<T>::value and
				not std::is_same<Lambda2, std::nullptr_t>::value,
			void>
		execute_function(
			T value, Lambda1 scalar_func, Lambda2 vector_func,
			Lambda3 matrix_func)
		{
			vector_func(value);
		}

		template <
			typename T, typename Lambda1, typename Lambda2, typename Lambda3>
		typename std::enable_if_t<
			is_matrix<T>::value and
				not std::is_same<Lambda3, std::nullptr_t>::value,
			void>
		execute_function(
			T value, Lambda1 scalar_func, Lambda2 vector_func,
			Lambda3 matrix_func)
		{
			matrix_func(value);
		}

		template <
			typename T, typename Lambda1, typename Lambda2, typename Lambda3>
		typename std::enable_if_t<
			(std::is_same<Lambda1, std::nullptr_t>::value and
			 std::is_scalar<T>::value) or
				(std::is_same<Lambda2, std::nullptr_t>::value and
				 is_vector<T>::value) or
				(std::is_same<Lambda3, std::nullptr_t>::value and
				 is_matrix<T>::value),
			void>
		execute_function(
			T value, Lambda1 scalar_func, Lambda2 vector_func,
			Lambda3 matrix_func)
		{
			SG_SWARNING(
				"Ignoring Any dispatch call.\n"
				"sg_for_each_type requires a lambda function definition "
				"for the expected underlying type of Any (%s).\n",
				demangled_type<T>().c_str())
		}

		template <
			typename TypeList, typename Lambda1, typename Lambda2,
			typename Lambda3>
		typename std::enable_if<
			std::is_same<TypeList, Types0>::value, void>::type
		sg_type_finder(
			const Any& any, TYPE type, Lambda1 scalar_func, Lambda2 vector_func,
			Lambda3 matrix_func)
		{
			SG_SERROR(
				"Unsupported type %s\n",
				demangled_type(any.type_info().name()).c_str())
		}

		template <
			typename TypeList, typename Lambda1, typename Lambda2,
			typename Lambda3>
		typename std::enable_if<
			(not std::is_same<TypeList, Types0>::value), void>::type
		sg_type_finder(
			const Any& any, TYPE type, Lambda1 scalar_func, Lambda2 vector_func,
			Lambda3 matrix_func)
		{
			if (type == sg_primitive_type<typename TypeList::Head>::pvalue)
			{
				typename TypeList::Head temporary_type_holder;
				execute_function(
					temporary_type_holder, scalar_func, vector_func,
					matrix_func);
			}
			else
				sg_type_finder<typename TypeList::Tail>(
					any, type, scalar_func, vector_func, matrix_func);
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
			ADD_TYPE_TO_MAP(SGVector<float64_t>, TYPE::PT_SGVECTOR_FLOAT64)
			ADD_TYPE_TO_MAP(SGVector<floatmax_t>, TYPE::PT_SGVECTOR_FLOATMAX)
			ADD_TYPE_TO_MAP(SGVector<int32_t>, TYPE::PT_SGVECTOR_INT32)
			ADD_TYPE_TO_MAP(SGVector<int64_t>, TYPE::PT_SGVECTOR_INT64)
			ADD_TYPE_TO_MAP(SGMatrix<float32_t>, TYPE::PT_SGMATRIX_FLOAT32)
			ADD_TYPE_TO_MAP(SGMatrix<float64_t>, TYPE::PT_SGMATRIX_FLOAT64)
			ADD_TYPE_TO_MAP(SGMatrix<floatmax_t>, TYPE::PT_SGMATRIX_FLOATMAX)
			ADD_TYPE_TO_MAP(SGMatrix<int32_t>, TYPE::PT_SGMATRIX_INT32)
			ADD_TYPE_TO_MAP(SGMatrix<int64_t>, TYPE::PT_SGMATRIX_INT64)
	};
	typemap sg_vector_types = {
			ADD_TYPE_TO_MAP(SGVector<float32_t>, TYPE::PT_SGVECTOR_FLOAT32)
			ADD_TYPE_TO_MAP(SGVector<float64_t>, TYPE::PT_SGVECTOR_FLOAT64)
			ADD_TYPE_TO_MAP(SGVector<floatmax_t>, TYPE::PT_SGVECTOR_FLOATMAX)
			ADD_TYPE_TO_MAP(SGVector<int32_t>, TYPE::PT_SGVECTOR_INT32)
			ADD_TYPE_TO_MAP(SGVector<int64_t>, TYPE::PT_SGVECTOR_INT64)
	};
	typemap sg_matrix_types = {
			ADD_TYPE_TO_MAP(SGMatrix<float32_t>, TYPE::PT_SGMATRIX_FLOAT32)
			ADD_TYPE_TO_MAP(SGMatrix<float64_t>, TYPE::PT_SGMATRIX_FLOAT64)
			ADD_TYPE_TO_MAP(SGMatrix<floatmax_t>, TYPE::PT_SGMATRIX_FLOATMAX)
			ADD_TYPE_TO_MAP(SGMatrix<int32_t>, TYPE::PT_SGMATRIX_INT32)
			ADD_TYPE_TO_MAP(SGMatrix<int64_t>, TYPE::PT_SGMATRIX_INT64)
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
}; // namespace shogun
#undef ADD_TYPE_TO_MAP

/** Checks the underlying type of an Any instance and
 * executes the appropriate lambda. The type is checked
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
 * The function accepts three lambda expressions for the cases where the type
 * is std::scalar, SGVector or SGMatrix.
 *
 * @param any Any object
 * @param typesmap check the underlying type of Any given this map
 * @param scalar_func lambda to execute if underlying type is a std::scalar
 * @param vector_func lambda to execute if underlying type is a SGVector
 * @param matrix_func lambda to execute if underlying type is a SGMatrix
 */
template <
	typename Lambda1 = std::nullptr_t, typename Lambda2 = std::nullptr_t,
	typename Lambda3 = std::nullptr_t>
void sg_for_each_type(
	const Any& any, const typemap& typesmap, Lambda1 scalar_func = nullptr,
	Lambda2 vector_func = nullptr, Lambda3 matrix_func = nullptr)
{
	TYPE type = type_internal::get_type(any, typesmap);
	if (type == TYPE::PT_UNDEFINED)
		SG_SERROR(
			"Type %s is not part of %s\n",
			demangled_type(any.type_info().name()).c_str(),
			type_internal::print_map(typesmap).c_str())
	else
		type_internal::sg_type_finder<SG_TYPES>(
			any, type, scalar_func, vector_func, matrix_func);
}

} // namespace shogun

#endif // SHOGUN_TYPE_CASE_H
