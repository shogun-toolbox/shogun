/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann
 */

#ifndef BASE_TYPES_H
#define BASE_TYPES_H

#include <shogun/lib/common.h>

namespace shogun
{

	// all shogun base classes for put/add templates
	class CMachine;
	class CKernel;
	class CDistance;
	class CFeatures;
	class CLabels;
	class CECOCEncoder;
	class CECOCDecoder;
	class CEvaluation;
	class CEvaluationResult;
	class CMulticlassStrategy;
	class CNeuralLayer;
	class CSplittingStrategy;
	class CPipeline;
	class CSVM;
	class CLikelihoodModel;
	class CMeanFunction;
	class CDifferentiableFunction;
	class CInference;
	class CLossFunction;
	class CTokenizer;

	// type trait to enable certain methods only for shogun base types
	template <class T>
	struct is_sg_base
	    : std::integral_constant<
	          bool, std::is_same<CMachine, T>::value ||
	                    std::is_same<CKernel, T>::value ||
	                    std::is_same<CDistance, T>::value ||
	                    std::is_same<CFeatures, T>::value ||
	                    std::is_same<CLabels, T>::value ||
	                    std::is_same<CECOCEncoder, T>::value ||
	                    std::is_same<CECOCDecoder, T>::value ||
	                    std::is_same<CEvaluation, T>::value ||
	                    std::is_same<CMulticlassStrategy, T>::value ||
	                    std::is_same<CNeuralLayer, T>::value ||
	                    std::is_same<CSplittingStrategy, T>::value ||
	                    std::is_same<CSVM, T>::value ||
	                    std::is_same<CDifferentiableFunction, T>::value ||
	                    std::is_same<CInference, T>::value ||
	                    std::is_same<CLikelihoodModel, T>::value ||
	                    std::is_same<CMeanFunction, T>::value ||
	                    std::is_same<CLossFunction, T>::value ||
	                    std::is_same<CTokenizer, T>::value ||
	                    std::is_same<CEvaluationResult, T>::value>
	{
	};

	template <class T>
	struct is_string
	    : std::integral_constant<
	          bool,
	          std::is_same<std::string, typename std::decay<T>::type>::value ||
	              std::is_same<char*, typename std::decay<T>::type>::value ||
	              std::is_same<
	                  const char*, typename std::decay<T>::type>::value>
	{
	};

	// Primitive types
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


	template <class T> class SGVector;
	template <class T> class SGMatrix;

	namespace type_internal
	{
		template <typename T>
		struct sg_type
		{
		};

		template <typename T>
		struct is_sg_primitive : public std::false_type
		{
		};

		template <typename T>
		struct is_sg_vector : public std::false_type
		{
		};

		template <typename T>
		struct is_sg_matrix : public std::false_type
		{
		};

#define SG_ADD_TYPE(T, type_)                                                  \
	template <>                                                                \
	struct sg_type<T>                                                          \
	{                                                                          \
		static constexpr TYPE ptype = type_;                                   \
	};
#define SG_ADD_PRIMITIVE_TYPE(T, type_)                                        \
	SG_ADD_TYPE(T, type_)                                                      \
	template <>                                                                \
	struct is_sg_primitive<T> : public std::true_type                          \
	{                                                                          \
	};
#define SG_ADD_SGVECTOR_TYPE(T, type_)                                         \
	SG_ADD_TYPE(T, type_)                                                      \
	template <>                                                                \
	struct is_sg_vector<T> : public std::true_type                             \
	{                                                                          \
	};
#define SG_ADD_SGMATRIX_TYPE(T, type_)                                         \
	SG_ADD_TYPE(T, type_)                                                      \
	template <>                                                                \
	struct is_sg_matrix<T> : public std::true_type                             \
	{                                                                          \
	};

		SG_ADD_PRIMITIVE_TYPE(bool, TYPE::T_BOOL)
		SG_ADD_PRIMITIVE_TYPE(char, TYPE::T_CHAR)
		SG_ADD_PRIMITIVE_TYPE(int8_t, TYPE::T_INT8)
		SG_ADD_PRIMITIVE_TYPE(uint8_t, TYPE::T_UINT8)
		SG_ADD_PRIMITIVE_TYPE(int16_t, TYPE::T_INT16)
		SG_ADD_PRIMITIVE_TYPE(uint16_t, TYPE::T_UINT16)
		SG_ADD_PRIMITIVE_TYPE(int32_t, TYPE::T_INT32)
		SG_ADD_PRIMITIVE_TYPE(uint32_t, TYPE::T_UINT32)
		SG_ADD_PRIMITIVE_TYPE(int64_t, TYPE::T_INT64)
		SG_ADD_PRIMITIVE_TYPE(uint64_t, TYPE::T_UINT64)
		SG_ADD_PRIMITIVE_TYPE(float32_t, TYPE::T_FLOAT32)
		SG_ADD_PRIMITIVE_TYPE(float64_t, TYPE::T_FLOAT64)
		SG_ADD_PRIMITIVE_TYPE(floatmax_t, TYPE::T_FLOATMAX)
		SG_ADD_PRIMITIVE_TYPE(complex128_t, TYPE::T_COMPLEX128)
SG_ADD_SGVECTOR_TYPE(SGVector<float32_t>, TYPE::T_SGVECTOR_FLOAT32)
SG_ADD_SGVECTOR_TYPE(SGVector<float64_t>, TYPE::T_SGVECTOR_FLOAT64)
SG_ADD_SGVECTOR_TYPE(SGVector<floatmax_t>, TYPE::T_SGVECTOR_FLOATMAX)
SG_ADD_SGVECTOR_TYPE(SGVector<int32_t>, TYPE::T_SGVECTOR_INT32)
SG_ADD_SGVECTOR_TYPE(SGVector<int64_t>, TYPE::T_SGVECTOR_INT64)
SG_ADD_SGMATRIX_TYPE(SGMatrix<float32_t>, TYPE::T_SGMATRIX_FLOAT32)
SG_ADD_SGMATRIX_TYPE(SGMatrix<float64_t>, TYPE::T_SGMATRIX_FLOAT64)
SG_ADD_SGMATRIX_TYPE(SGMatrix<floatmax_t>, TYPE::T_SGMATRIX_FLOATMAX)
SG_ADD_SGMATRIX_TYPE(SGMatrix<int32_t>, TYPE::T_SGMATRIX_INT32)
SG_ADD_SGMATRIX_TYPE(SGMatrix<int64_t>, TYPE::T_SGMATRIX_INT64)

#undef SG_ADD_TYPE
#undef SG_ADD_PRIMITIVE_TYPE
#undef SG_ADD_SGVECTOR_TYPE
#undef SG_ADD_SGMATRIX_TYPE
	}


} // namespace shogun

#endif // BASE_TYPES__H
