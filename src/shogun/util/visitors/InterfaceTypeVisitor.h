#ifndef _BASE_TYPE_VISITOR_H_
#define _BASE_TYPE_VISITOR_H_

#include <shogun/base/base_types.h>

#include <memory>

namespace shogun
{
	class SGObject;

	class InterfaceTypeVisitor
	{
	public:
		virtual ~InterfaceTypeVisitor() = default;

		virtual void on(std::shared_ptr<SGObject>*) = 0;
		virtual void on(std::shared_ptr<Machine>*) = 0;
		virtual void on(std::shared_ptr<Kernel>*) = 0;
		virtual void on(std::shared_ptr<Distance>*) = 0;
		virtual void on(std::shared_ptr<Features>*) = 0;
		virtual void on(std::shared_ptr<Labels>*) = 0;
		virtual void on(std::shared_ptr<ECOCEncoder>*) = 0;
		virtual void on(std::shared_ptr<ECOCDecoder>*) = 0;
		virtual void on(std::shared_ptr<Evaluation>*) = 0;
		virtual void on(std::shared_ptr<EvaluationResult>*) = 0;
		virtual void on(std::shared_ptr<MulticlassStrategy>*) = 0;
		virtual void on(std::shared_ptr<NeuralLayer>*) = 0;
		virtual void on(std::shared_ptr<SplittingStrategy>*) = 0;
		virtual void on(std::shared_ptr<Pipeline>*) = 0;
		virtual void on(std::shared_ptr<SVM>*) = 0;
		virtual void on(std::shared_ptr<LikelihoodModel>*) = 0;
		virtual void on(std::shared_ptr<MeanFunction>*) = 0;
		virtual void on(std::shared_ptr<DifferentiableFunction>*) = 0;
		virtual void on(std::shared_ptr<Inference>*) = 0;
		virtual void on(std::shared_ptr<LossFunction>*) = 0;
		virtual void on(std::shared_ptr<Tokenizer>*) = 0;
		virtual void on(std::shared_ptr<CombinationRule>*) = 0;
		virtual void on(std::shared_ptr<KernelNormalizer>*) = 0;
		virtual void on(std::shared_ptr<Transformer>*) = 0;
		virtual void on(std::shared_ptr<MachineEvaluation>*) = 0;
		virtual void on(std::shared_ptr<StructuredModel>*) = 0;
		virtual void on(std::shared_ptr<FactorType>*) = 0;
		virtual void on(std::shared_ptr<ParameterObserver>*) = 0;
		virtual void on(std::shared_ptr<Distribution>*) = 0;
		virtual void on(std::shared_ptr<GaussianProcess>*) = 0;
		virtual void on(std::shared_ptr<Alphabet>*) = 0;

		template <
		    class T,
		    std::enable_if_t<std::is_base_of_v<SGObject, T>, T>* = nullptr>
		void on(std::shared_ptr<T>* v)
		{
			if (!v)
				return;

			using Base = std::conditional_t<
			    std::is_same_v<base_type<T>, std::nullptr_t>, SGObject,
			    base_type<T>>;

			auto v_upcasted = std::static_pointer_cast<Base>(*v);
			on(&v_upcasted);
		}

		template <
		    class T,
		    std::enable_if_t<!std::is_base_of_v<SGObject, T>, T>* = nullptr>
		void on(std::shared_ptr<T>* v)
		{
		}
	};
} // namespace shogun

#endif
