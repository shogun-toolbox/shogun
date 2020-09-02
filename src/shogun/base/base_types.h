/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann
 */

#ifndef BASE_TYPES_H
#define BASE_TYPES_H

#include <memory>

namespace shogun
{

	// all shogun base classes for put/add templates and factories
	class Machine;
	class Kernel;
	class Distance;
	class Features;
	class Labels;
	class ECOCEncoder;
	class ECOCDecoder;
	class Evaluation;
	class EvaluationResult;
	class MulticlassStrategy;
	class NeuralLayer;
	class SplittingStrategy;
	class Pipeline;
	class SVM;
	class LikelihoodModel;
	class MeanFunction;
	class DifferentiableFunction;
	class Inference;
	class LossFunction;
	class Tokenizer;
	class CombinationRule;
	class KernelNormalizer;
	class Transformer;
	class MachineEvaluation;
	class StructuredModel;
	class FactorType;
	class ParameterObserver;
	class Distribution;
	class GaussianProcess;
	class Alphabet;

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

	using sg_inferface = type_list<Machine, Kernel, Distance,
		Features, Labels, ECOCEncoder, ECOCDecoder, Evaluation,
		EvaluationResult, MulticlassStrategy, NeuralLayer,
		SplittingStrategy, LikelihoodModel, MeanFunction,
		DifferentiableFunction, Inference, LossFunction,
		Tokenizer, CombinationRule, KernelNormalizer, Transformer,
		MachineEvaluation, StructuredModel, FactorType, ParameterObserver,
		Distribution, GaussianProcess, Alphabet>;

	namespace types_detail
	{
		template <typename T, typename... Ts>
		struct typeInList_impl : public std::false_type
		{
		};

		template <typename T, typename T1, typename... Ts>
		struct typeInList_impl<T, T1, Ts...>
		    : public std::conditional_t<
		          std::is_same_v<T, T1>, std::true_type,
		          typeInList_impl<T, Ts...>>
		{
		};
	} // namespace types_detail

	template <typename T, typename TypesT>
	struct typeInList : public std::false_type
	{
		using X = typename TypesT : WTF;
	};

	template <typename T, template <typename...> class TypesT, typename... Args>
	struct typeInList<T, TypesT<Args...>>
	    : public types_detail::typeInList_impl<T, Args...>
	{
	};


	template <class T>
	struct is_sg_base : public typeInList<T, sg_inferface>
	{
	};

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

	template <class T>
   	struct remove_shared_ptr
	{
		using type = T;
	};

	template <class T>
	struct remove_shared_ptr<std::shared_ptr<T>>
    	{
        	using type = T;
	};

	template <class T>
	using remove_shared_ptr_t = typename remove_shared_ptr<T>::type;

	struct AutoValueEmpty
	{
		bool operator==(const AutoValueEmpty& other) const
		{
			return true;
		}
	};

	template <typename T>
	using AutoValue = std::variant<T, AutoValueEmpty>;

	template <typename T>
	struct is_auto_value: std::false_type {};

	template <typename T>
	struct is_auto_value<AutoValue<T>>: std::true_type {};

	template <typename T>
	inline constexpr bool is_auto_value_v = is_auto_value<T>::value;
} // namespace shogun

#endif // BASE_TYPES__H
