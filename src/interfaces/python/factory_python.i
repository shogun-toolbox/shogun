%{
#include <shogun/util/visitors/InterfaceTypeVisitor.h>

// we can use:
// SWIG_TypeQueryModule or SWIG_MangledTypeQueryModule or SWIG_Python_TypeQuery
// to query the swig_type and be less implementation dependent
// but this is faster and should not cause problems
#define SHOGUN_GET_SWIG_TYPE(name)                                             \
	SWIGTYPE_p_std__shared_ptrT_shogun__##name##_t

namespace shogun
{
	class ShogunInterfaceToPyObject : public InterfaceTypeVisitor
	{
	public:
		virtual void on(std::shared_ptr<SGObject>* v)
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(SGObject));
		}
		virtual void on(std::shared_ptr<Machine>* v)
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(Machine));
		}
		virtual void on(std::shared_ptr<Kernel>* v)
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(Kernel));
		}
		virtual void on(std::shared_ptr<Distance>* v)
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(Distance));
		}
		virtual void on(std::shared_ptr<Features>* v)
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(Features));
		}
		virtual void on(std::shared_ptr<Labels>* v)
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(Labels));
		}
		virtual void on(std::shared_ptr<ECOCEncoder>* v)
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(ECOCEncoder));
		}
		virtual void on(std::shared_ptr<ECOCDecoder>* v)
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(ECOCDecoder));
		}
		virtual void on(std::shared_ptr<Evaluation>* v)
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(Evaluation));
		}
		virtual void on(std::shared_ptr<EvaluationResult>* v)
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(EvaluationResult));
		}
		virtual void on(std::shared_ptr<MulticlassStrategy>* v)
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(MulticlassStrategy));
		}
		virtual void on(std::shared_ptr<NeuralLayer>* v)
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(NeuralLayer));
		}
		virtual void on(std::shared_ptr<SplittingStrategy>* v)
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(SplittingStrategy));
		}
		virtual void on(std::shared_ptr<Pipeline>* v)
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(Pipeline));
		}
		virtual void on(std::shared_ptr<SVM>* v)
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(SVM));
		}
		virtual void on(std::shared_ptr<LikelihoodModel>* v)
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(LikelihoodModel));
		}
		virtual void on(std::shared_ptr<MeanFunction>* v)
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(MeanFunction));
		}
		virtual void on(std::shared_ptr<DifferentiableFunction>* v)
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(DifferentiableFunction));
		}
		virtual void on(std::shared_ptr<Inference>* v)
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(Inference));
		}
		virtual void on(std::shared_ptr<LossFunction>* v)
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(LossFunction));
		}
		virtual void on(std::shared_ptr<Tokenizer>* v)
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(Tokenizer));
		}
		virtual void on(std::shared_ptr<CombinationRule>* v)
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(CombinationRule));
		}
		virtual void on(std::shared_ptr<KernelNormalizer>* v)
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(KernelNormalizer));
		}
		virtual void on(std::shared_ptr<Transformer>* v)
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(Transformer));
		}
		virtual void on(std::shared_ptr<MachineEvaluation>* v)
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(MachineEvaluation));
		}
		virtual void on(std::shared_ptr<StructuredModel>* v)
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(StructuredModel));
		}
		virtual void on(std::shared_ptr<FactorType>* v)
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(FactorType));
		}
		virtual void on(std::shared_ptr<ParameterObserver>* v)
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(ParameterObserver));
		}
		virtual void on(std::shared_ptr<Distribution>* v)
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(Distribution));
		}
		virtual void on(std::shared_ptr<GaussianProcess>* v)
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(GaussianProcess));
		}
		virtual void on(std::shared_ptr<Alphabet>* v)
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(Alphabet));
		}

		template <typename T>
		void on_impl(std::shared_ptr<T>* v, swig_type_info* type)
		{
			if (!v)
				return;

			auto smartresult = new std::shared_ptr<T>(*v);
			m_pyobj = SWIG_Python_NewPointerObj(
				nullptr, SWIG_as_voidptr(smartresult), type, SWIG_POINTER_OWN);
		}

		PyObject* pyobj()
		{
			if (!m_pyobj)
			{
				auto smartresult = new std::shared_ptr<SGObject>(nullptr);
				m_pyobj = SWIG_Python_NewPointerObj(
					nullptr, SWIG_as_voidptr(smartresult),
					SHOGUN_GET_SWIG_TYPE(SGObject), SWIG_POINTER_OWN);
			}
			return m_pyobj;
		}

	private:
		PyObject* m_pyobj = nullptr;
	};
} // namespace shogun
%}

%inline %{
#include <shogun/base/class_list.h>

namespace shogun
{
	PyObject* create(const char* name)
	{
		static auto visitor = std::make_shared<ShogunInterfaceToPyObject>();
		create(name, PT_NOT_GENERIC, visitor);
		return visitor->pyobj();
	}
}

%}
