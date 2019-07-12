/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann
 */

#ifndef BASE_TYPES_H
#define BASE_TYPES_H

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
	// FIXME: use sg_interface to populate this trait
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

	// General type traits to recognize SGMatrix and SGVectors.
	template <typename T>
	class SGMatrix;
	template <typename T>
	class SGVector;

	template <typename>
	struct is_sg_vector : std::false_type
	{
	};

	template <typename T>
	struct is_sg_vector<SGVector<T>> : std::true_type
	{
	};

	template <typename>
	struct is_sg_matrix : std::false_type
	{
	};

	template <typename T>
	struct is_sg_matrix<SGMatrix<T>> : std::true_type
	{
	};

	template <typename...>
	struct type_list{};

	template <typename T>
	struct type_holder
	{
		typedef T type;
	};

	using sg_inferface = type_list<CMachine, CKernel, CDistance,
		CFeatures, CLabels, CECOCEncoder, CECOCDecoder, CEvaluation,
		CEvaluationResult, CMulticlassStrategy, CNeuralLayer,
		CSplittingStrategy, CLikelihoodModel, CMeanFunction,
		CDifferentiableFunction, CInference, CLossFunction,
		CTokenizer>;

	template <typename Derived>
	constexpr auto find_base(type_list<>)
	{
		return type_holder<std::nullptr_t>{};
	}

	template <typename Derived, typename T, typename... Ts>
	constexpr auto find_base(type_list<T, Ts...>) {
	    if constexpr (std::is_base_of_v<T, Derived>)
	        return type_holder<T>{};
	    else
	        return find_base<Derived>(type_list<Ts...>{});
	}

	template <typename Derived>
	using base_type = typename decltype(find_base<Derived>(sg_inferface{}))::type;

} // namespace shogun

#endif // BASE_TYPES__H
