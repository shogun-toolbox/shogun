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
		T_BOOL = 1,
		T_CHAR = 2,
		T_INT8 = 3,
		T_UINT8 = 4,
		T_INT16 = 5,
		T_UINT16 = 6,
		T_INT32 = 7,
		T_UINT32 = 8,
		T_INT64 = 9,
		T_UINT64 = 10,
		T_FLOAT32 = 11,
		T_FLOAT64 = 12,
		T_FLOATMAX = 13,
		T_SGOBJECT = 14,
		T_COMPLEX128 = 15,
		T_SGVECTOR_FLOAT32 = 16,
		T_SGVECTOR_FLOAT64 = 17,
		T_SGVECTOR_FLOATMAX = 18,
		T_SGVECTOR_INT32 = 19,
		T_SGVECTOR_INT64 = 20,
		T_SGMATRIX_FLOAT32 = 21,
		T_SGMATRIX_FLOAT64 = 22,
		T_SGMATRIX_FLOATMAX = 23,
		T_SGMATRIX_INT32 = 24,
		T_SGMATRIX_INT64 = 25,
		T_UNDEFINED = 26
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

SG_PRIMITIVE_TYPE(bool, TYPE::T_BOOL)
SG_PRIMITIVE_TYPE(char, TYPE::T_CHAR)
SG_PRIMITIVE_TYPE(int8_t, TYPE::T_INT8)
SG_PRIMITIVE_TYPE(uint8_t, TYPE::T_UINT8)
SG_PRIMITIVE_TYPE(int16_t, TYPE::T_INT16)
SG_PRIMITIVE_TYPE(uint16_t, TYPE::T_UINT16)
SG_PRIMITIVE_TYPE(int32_t, TYPE::T_INT32)
SG_PRIMITIVE_TYPE(uint32_t, TYPE::T_UINT32)
SG_PRIMITIVE_TYPE(int64_t, TYPE::T_INT64)
SG_PRIMITIVE_TYPE(uint64_t, TYPE::T_UINT64)
SG_PRIMITIVE_TYPE(float32_t, TYPE::T_FLOAT32)
SG_PRIMITIVE_TYPE(float64_t, TYPE::T_FLOAT64)
SG_PRIMITIVE_TYPE(floatmax_t, TYPE::T_FLOATMAX)
SG_PRIMITIVE_TYPE(complex128_t, TYPE::T_COMPLEX128)
SG_PRIMITIVE_TYPE(SGVector<float32_t>, TYPE::T_SGVECTOR_FLOAT32)
SG_PRIMITIVE_TYPE(SGVector<float64_t>, TYPE::T_SGVECTOR_FLOAT64)
SG_PRIMITIVE_TYPE(SGVector<floatmax_t>, TYPE::T_SGVECTOR_FLOATMAX)
SG_PRIMITIVE_TYPE(SGVector<int32_t>, TYPE::T_SGVECTOR_INT32)
SG_PRIMITIVE_TYPE(SGVector<int64_t>, TYPE::T_SGVECTOR_INT64)
SG_PRIMITIVE_TYPE(SGMatrix<float32_t>, TYPE::T_SGMATRIX_FLOAT32)
SG_PRIMITIVE_TYPE(SGMatrix<float64_t>, TYPE::T_SGMATRIX_FLOAT64)
SG_PRIMITIVE_TYPE(SGMatrix<floatmax_t>, TYPE::T_SGMATRIX_FLOATMAX)
SG_PRIMITIVE_TYPE(SGMatrix<int32_t>, TYPE::T_SGMATRIX_INT32)
SG_PRIMITIVE_TYPE(SGMatrix<int64_t>, TYPE::T_SGMATRIX_INT64)

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

			return it == map.end() ? TYPE::T_UNDEFINED : map.at(type);
		}

		template <typename T>
		struct wrong : std::false_type
		{
		};

		struct ok
		{
		};

		struct assert_return_type_is_valid
		{
			template <typename T = void>
			assert_return_type_is_valid()
			{
				static_assert(
				    wrong<T>::value, "All lambda definitions must be void and "
				                     "have the signature 'void f(auto type)'");
			}
		};

		template <typename T>
		struct is_sg_vector : public std::false_type
		{
		};
		template <typename T>
		struct is_sg_vector<SGVector<T>> : public std::true_type
		{
		};

		template <typename T>
		struct is_sg_matrix : public std::false_type
		{
		};
		template <typename T>
		struct is_sg_matrix<SGMatrix<T>> : public std::true_type
		{
		};

		template <typename T>
		struct is_none : public std::false_type
		{
		};
		template <>
		struct is_none<None> : public std::true_type
		{
		};

		template <bool Condition, typename T = void, typename FuncT = void>
		struct check_void_if
		{
		};

		template <typename T, typename FunctorT>
		struct check_void_if<false, T, FunctorT>
		{
			using type = std::true_type;
			static constexpr bool value = true;
		};

		template <typename T, typename FunctorT>
		struct check_void_if<true, T, FunctorT>
		{
			using type = typename std::is_void<
			    typename std::result_of_t<FunctorT(T)>>::type;
			static constexpr bool value =
			    std::is_void<typename std::result_of_t<FunctorT(T)>>::value;
		};

		template <
		    typename T, typename ScalarLambdaT, typename VectorLambdaT,
		    typename MatrixLambdaT>
		using check_return_type = std::conditional_t<
		    check_void_if<
		        not is_none<ScalarLambdaT>::value and std::is_scalar<T>::value,
		        T, ScalarLambdaT>::value and
		        check_void_if<
		            not is_none<VectorLambdaT>::value and
		                is_sg_vector<T>::value,
		            T, VectorLambdaT>::value and
		        check_void_if<
		            not is_none<MatrixLambdaT>::value and
		                is_sg_matrix<T>::value,
		            T, MatrixLambdaT>::value,
		    ok, assert_return_type_is_valid>;

		template <
		    typename T, typename ScalarLambdaT, typename VectorLambdaT,
		    typename MatrixLambdaT,
		    typename std::enable_if_t<
		        std::is_scalar<T>::value and
		        not is_none<ScalarLambdaT>::value>* = nullptr>
		auto execute_function(
		    ok, T value, ScalarLambdaT scalar_func, VectorLambdaT vector_func,
		    MatrixLambdaT matrix_func) -> void
		{
			scalar_func(value);
		}

		template <
		    typename Check, typename T, typename ScalarLambdaT,
		    typename VectorLambdaT, typename MatrixLambdaT,
		    typename std::enable_if_t<
		        std::is_scalar<T>::value and
		        not is_none<ScalarLambdaT>::value>* = nullptr>
		auto execute_function(
		    Check, T value, ScalarLambdaT scalar_func,
		    VectorLambdaT vector_func, MatrixLambdaT matrix_func) -> Check
		{
		}

		template <
		    typename T, typename ScalarLambdaT, typename VectorLambdaT,
		    typename MatrixLambdaT,
		    typename std::enable_if_t<
		        is_sg_vector<T>::value and not is_none<VectorLambdaT>::value>* =
		        nullptr>
		auto execute_function(
		    ok, T value, ScalarLambdaT scalar_func, VectorLambdaT vector_func,
		    MatrixLambdaT matrix_func) -> void
		{
			vector_func(value);
		}

		template <
		    typename Check, typename T, typename ScalarLambdaT,
		    typename VectorLambdaT, typename MatrixLambdaT,
		    typename std::enable_if_t<
		        is_sg_vector<T>::value and not is_none<MatrixLambdaT>::value>* =
		        nullptr>
		auto execute_function(
		    Check, T value, ScalarLambdaT scalar_func,
		    VectorLambdaT vector_func, MatrixLambdaT matrix_func) -> Check
		{
		}

		template <
		    typename T, typename ScalarLambdaT, typename VectorLambdaT,
		    typename MatrixLambdaT,
		    typename std::enable_if_t<
		        is_sg_matrix<T>::value and not is_none<MatrixLambdaT>::value>* =
		        nullptr>
		auto execute_function(
		    ok, T value, ScalarLambdaT scalar_func, VectorLambdaT vector_func,
		    MatrixLambdaT matrix_func) -> void
		{
			matrix_func(value);
		}

		template <
		    typename Check, typename T, typename ScalarLambdaT,
		    typename VectorLambdaT, typename MatrixLambdaT,
		    typename std::enable_if_t<
		        is_sg_matrix<T>::value and not is_none<MatrixLambdaT>::value>* =
		        nullptr>
		auto execute_function(
		    Check, T value, ScalarLambdaT scalar_func,
		    VectorLambdaT vector_func, MatrixLambdaT matrix_func) -> Check
		{
		}

		template <
		    typename T, typename ScalarLambdaT, typename VectorLambdaT,
		    typename MatrixLambdaT,
		    typename std::enable_if_t<
		        (is_none<ScalarLambdaT>::value and std::is_scalar<T>::value) or
		        (is_none<VectorLambdaT>::value and is_sg_vector<T>::value) or
		        (is_none<MatrixLambdaT>::value and is_sg_matrix<T>::value)>* =
		        nullptr>
		auto execute_function(
		    ok, T value, ScalarLambdaT scalar_func, VectorLambdaT vector_func,
		    MatrixLambdaT matrix_func) -> void
		{
			SG_SWARNING(
			    "Ignoring Any dispatch call.\n"
			    "sg_any_dispatch requires a lambda function definition "
			    "for the expected underlying type of Any (%s).\n",
			    demangled_type<T>().c_str())
		}

		template <
		    typename Check, typename T, typename ScalarLambdaT,
		    typename VectorLambdaT, typename MatrixLambdaT,
		    typename std::enable_if_t<
		        (is_none<ScalarLambdaT>::value and std::is_scalar<T>::value) or
		        (is_none<VectorLambdaT>::value and is_sg_vector<T>::value) or
		        (is_none<MatrixLambdaT>::value and is_sg_matrix<T>::value)>* =
		        nullptr>
		auto execute_function(
		    Check, T value, ScalarLambdaT scalar_func,
		    VectorLambdaT vector_func, MatrixLambdaT matrix_func) -> Check
		{
		}

		template <
		    typename TypeList, typename ScalarLambdaT, typename VectorLambdaT,
		    typename MatrixLambdaT>
		typename std::enable_if<
		    std::is_same<TypeList, Types0>::value, void>::type
		sg_type_finder(
		    const Any& any, TYPE type, ScalarLambdaT scalar_func,
		    VectorLambdaT vector_func, MatrixLambdaT matrix_func)
		{
			SG_SERROR(
			    "Unsupported type %s\n",
			    demangled_type(any.type_info().name()).c_str())
		}

		template <
		    typename TypeList, typename ScalarLambdaT, typename VectorLambdaT,
		    typename MatrixLambdaT,
		    typename std::enable_if<
		        not std::is_same<TypeList, Types0>::value>::type* = nullptr>
		auto sg_type_finder(
		    const Any& any, TYPE type, ScalarLambdaT scalar_func,
		    VectorLambdaT vector_func, MatrixLambdaT matrix_func)
		    -> check_return_type<
		        typename TypeList::Head, ScalarLambdaT, VectorLambdaT,
		        MatrixLambdaT>
		{
			if (type == sg_primitive_type<typename TypeList::Head>::pvalue)
			{
				typename TypeList::Head temporary_type_holder;

				execute_function(
				    check_return_type<
				        typename TypeList::Head, ScalarLambdaT, VectorLambdaT,
				        MatrixLambdaT>{},
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
			ADD_TYPE_TO_MAP(bool, TYPE::T_BOOL)
			ADD_TYPE_TO_MAP(char, TYPE::T_CHAR)
			ADD_TYPE_TO_MAP(int8_t, TYPE::T_INT8)
			ADD_TYPE_TO_MAP(uint8_t, TYPE::T_UINT8)
			ADD_TYPE_TO_MAP(int16_t , TYPE::T_INT16)
			ADD_TYPE_TO_MAP(uint16_t , TYPE::T_UINT16)
			ADD_TYPE_TO_MAP(int32_t , TYPE::T_INT32)
			ADD_TYPE_TO_MAP(uint32_t , TYPE::T_UINT32)
			ADD_TYPE_TO_MAP(int64_t , TYPE::T_INT64)
			ADD_TYPE_TO_MAP(uint64_t , TYPE::T_UINT64)
			ADD_TYPE_TO_MAP(float32_t , TYPE::T_FLOAT32)
			ADD_TYPE_TO_MAP(float64_t , TYPE::T_FLOAT64)
			ADD_TYPE_TO_MAP(floatmax_t , TYPE::T_FLOATMAX)
			ADD_TYPE_TO_MAP(complex128_t, TYPE::T_COMPLEX128)
			ADD_TYPE_TO_MAP(SGVector<float32_t>, TYPE::T_SGVECTOR_FLOAT32)
			ADD_TYPE_TO_MAP(SGVector<float64_t>, TYPE::T_SGVECTOR_FLOAT64)
			ADD_TYPE_TO_MAP(SGVector<floatmax_t>, TYPE::T_SGVECTOR_FLOATMAX)
			ADD_TYPE_TO_MAP(SGVector<int32_t>, TYPE::T_SGVECTOR_INT32)
			ADD_TYPE_TO_MAP(SGVector<int64_t>, TYPE::T_SGVECTOR_INT64)
			ADD_TYPE_TO_MAP(SGMatrix<float32_t>, TYPE::T_SGMATRIX_FLOAT32)
			ADD_TYPE_TO_MAP(SGMatrix<float64_t>, TYPE::T_SGMATRIX_FLOAT64)
			ADD_TYPE_TO_MAP(SGMatrix<floatmax_t>, TYPE::T_SGMATRIX_FLOATMAX)
			ADD_TYPE_TO_MAP(SGMatrix<int32_t>, TYPE::T_SGMATRIX_INT32)
			ADD_TYPE_TO_MAP(SGMatrix<int64_t>, TYPE::T_SGMATRIX_INT64)
	};
	typemap sg_vector_types = {
			ADD_TYPE_TO_MAP(SGVector<float32_t>, TYPE::T_SGVECTOR_FLOAT32)
			ADD_TYPE_TO_MAP(SGVector<float64_t>, TYPE::T_SGVECTOR_FLOAT64)
			ADD_TYPE_TO_MAP(SGVector<floatmax_t>, TYPE::T_SGVECTOR_FLOATMAX)
			ADD_TYPE_TO_MAP(SGVector<int32_t>, TYPE::T_SGVECTOR_INT32)
			ADD_TYPE_TO_MAP(SGVector<int64_t>, TYPE::T_SGVECTOR_INT64)
	};
	typemap sg_matrix_types = {
			ADD_TYPE_TO_MAP(SGMatrix<float32_t>, TYPE::T_SGMATRIX_FLOAT32)
			ADD_TYPE_TO_MAP(SGMatrix<float64_t>, TYPE::T_SGMATRIX_FLOAT64)
			ADD_TYPE_TO_MAP(SGMatrix<floatmax_t>, TYPE::T_SGMATRIX_FLOATMAX)
			ADD_TYPE_TO_MAP(SGMatrix<int32_t>, TYPE::T_SGMATRIX_INT32)
			ADD_TYPE_TO_MAP(SGMatrix<int64_t>, TYPE::T_SGMATRIX_INT64)
	};
	typemap sg_non_complex_types = {
			ADD_TYPE_TO_MAP(bool, TYPE::T_BOOL)
			ADD_TYPE_TO_MAP(char, TYPE::T_CHAR)
			ADD_TYPE_TO_MAP(int8_t, TYPE::T_INT8)
			ADD_TYPE_TO_MAP(uint8_t, TYPE::T_UINT8)
			ADD_TYPE_TO_MAP(int16_t , TYPE::T_INT16)
			ADD_TYPE_TO_MAP(uint16_t , TYPE::T_UINT16)
			ADD_TYPE_TO_MAP(int32_t , TYPE::T_INT32)
			ADD_TYPE_TO_MAP(uint32_t , TYPE::T_UINT32)
			ADD_TYPE_TO_MAP(int64_t , TYPE::T_INT64)
			ADD_TYPE_TO_MAP(uint64_t , TYPE::T_UINT64)
			ADD_TYPE_TO_MAP(float32_t , TYPE::T_FLOAT32)
			ADD_TYPE_TO_MAP(float64_t , TYPE::T_FLOAT64)
			ADD_TYPE_TO_MAP(floatmax_t , TYPE::T_FLOATMAX)
	};
	typemap sg_real_types = {
			ADD_TYPE_TO_MAP(float32_t , TYPE::T_FLOAT32)
			ADD_TYPE_TO_MAP(float64_t , TYPE::T_FLOAT64)
			ADD_TYPE_TO_MAP(floatmax_t , TYPE::T_FLOATMAX)
	};
	typemap sg_non_integer_types = {
			ADD_TYPE_TO_MAP(float32_t , TYPE::T_FLOAT32)
			ADD_TYPE_TO_MAP(float64_t , TYPE::T_FLOAT64)
			ADD_TYPE_TO_MAP(floatmax_t , TYPE::T_FLOATMAX)
			ADD_TYPE_TO_MAP(complex128_t, TYPE::T_COMPLEX128)
	};
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
	 * The function accepts three lambda expressions for the cases where the
	 * type is scalar, SGVector or SGMatrix.
	 *
	 * @param any Any object
	 * @param typesmap check the underlying type of Any given this map
	 * @param scalar_func lambda to execute if underlying type is a scalar
	 * (std::is_scalar)
	 * @param vector_func lambda to execute if underlying type is a SGVector
	 * @param matrix_func lambda to execute if underlying type is a SGMatrix
	 */
	template <
	    typename ScalarLambdaT = None, typename VectorLambdaT = None,
	    typename MatrixLambdaT = None>
	auto sg_any_dispatch(
	    const Any& any, const typemap& typesmap,
	    ScalarLambdaT scalar_func = None{}, VectorLambdaT vector_func = None{},
	    MatrixLambdaT matrix_func = None{})
	    -> decltype(type_internal::sg_type_finder<SG_TYPES>(
	        any, type_internal::get_type(any, typesmap), scalar_func,
	        vector_func, matrix_func))
	{
		TYPE type = type_internal::get_type(any, typesmap);
		if (type == TYPE::T_UNDEFINED)
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
