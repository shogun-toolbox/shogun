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

	template <typename...>
	struct type_list{};

	using sg_inferface = type_list<Machine, Kernel, Distance,
		Features, Labels, ECOCEncoder, ECOCDecoder, Evaluation,
		EvaluationResult, MulticlassStrategy, NeuralLayer,
		SplittingStrategy, SVM, LikelihoodModel, MeanFunction,
		DifferentiableFunction, Inference, LossFunction,
		Tokenizer, CombinationRule>;

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

	template <typename T>
	struct type_holder
	{
		typedef T type;
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

	template <typename Derived>
	constexpr auto is_sg_base(type_list<>)
	{
		return std::false_type{};
	}

	template <typename Derived, typename T, typename... Ts>
	constexpr auto is_sg_base(type_list<T, Ts...>) {
		if constexpr (std::is_same_v<T, Derived>)
			return std::true_type{};
		else
			return is_sg_base<Derived>(type_list<Ts...>{});
	}

	template <typename Derived>
	inline constexpr bool is_sg_base_v = is_sg_base<remove_shared_ptr_t<Derived>>(sg_inferface{});

} // namespace shogun

#endif // BASE_TYPES__H
