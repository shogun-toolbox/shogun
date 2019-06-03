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

	// all shogun base classes for put/add templates
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

	// type trait to enable certain methods only for shogun base types
	// FIXME: use sg_interface to populate this trait
	template <class T>
	struct is_sg_base
	    : std::integral_constant<
	          bool, std::is_same<Machine, T>::value ||
	                    std::is_same<Kernel, T>::value ||
	                    std::is_same<Distance, T>::value ||
	                    std::is_same<Features, T>::value ||
	                    std::is_same<Labels, T>::value ||
	                    std::is_same<ECOCEncoder, T>::value ||
	                    std::is_same<ECOCDecoder, T>::value ||
	                    std::is_same<Evaluation, T>::value ||
	                    std::is_same<MulticlassStrategy, T>::value ||
	                    std::is_same<NeuralLayer, T>::value ||
	                    std::is_same<SplittingStrategy, T>::value ||
	                    std::is_same<SVM, T>::value ||
	                    std::is_same<DifferentiableFunction, T>::value ||
	                    std::is_same<Inference, T>::value ||
	                    std::is_same<LikelihoodModel, T>::value ||
	                    std::is_same<MeanFunction, T>::value ||
	                    std::is_same<LossFunction, T>::value ||
	                    std::is_same<Tokenizer, T>::value ||
	                    std::is_same<EvaluationResult, T>::value ||
	                    std::is_same<CombinationRule, T>::value>
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

	using sg_inferface = type_list<Machine, Kernel, Distance,
		Features, Labels, ECOCEncoder, ECOCDecoder, Evaluation,
		EvaluationResult, MulticlassStrategy, NeuralLayer,
		SplittingStrategy, LikelihoodModel, MeanFunction,
		DifferentiableFunction, Inference, LossFunction,
		Tokenizer>;

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

} // namespace shogun

#endif // BASE_TYPES__H
