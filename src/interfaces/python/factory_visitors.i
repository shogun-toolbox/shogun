%{
#include <shogun/util/visitors/InterfaceTypeVisitor.h>

// we can use:
// SWIG_TypeQueryModule or SWIG_MangledTypeQueryModule or SWIG_Python_TypeQuery
// to query the swig_type in run time and be less implementation dependent maybe?
// but this is faster (compile-time) and should not cause problems
#define SHOGUN_GET_SWIG_TYPE(name)                                             \
	SWIGTYPE_p_std__shared_ptrT_shogun__##name##_t

namespace shogun
{
	class ShogunInterfaceToPyObject : public InterfaceTypeVisitor
	{
	public:
		void on(std::shared_ptr<SGObject>* v) override
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(SGObject));
		}
		void on(std::shared_ptr<Machine>* v) override
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(Machine));
		}
		void on(std::shared_ptr<Kernel>* v) override
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(Kernel));
		}
		void on(std::shared_ptr<Distance>* v) override
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(Distance));
		}
		void on(std::shared_ptr<Features>* v) override
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(Features));
		}
		void on(std::shared_ptr<Labels>* v) override
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(Labels));
		}
		void on(std::shared_ptr<ECOCEncoder>* v) override
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(ECOCEncoder));
		}
		void on(std::shared_ptr<ECOCDecoder>* v) override
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(ECOCDecoder));
		}
		void on(std::shared_ptr<Evaluation>* v) override
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(Evaluation));
		}
		void on(std::shared_ptr<EvaluationResult>* v) override
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(EvaluationResult));
		}
		void on(std::shared_ptr<MulticlassStrategy>* v) override
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(MulticlassStrategy));
		}
		void on(std::shared_ptr<NeuralLayer>* v) override
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(NeuralLayer));
		}
		void on(std::shared_ptr<SplittingStrategy>* v) override
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(SplittingStrategy));
		}
		void on(std::shared_ptr<Pipeline>* v) override
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(Pipeline));
		}
		void on(std::shared_ptr<SVM>* v) override
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(SVM));
		}
		void on(std::shared_ptr<LikelihoodModel>* v) override
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(LikelihoodModel));
		}
		void on(std::shared_ptr<MeanFunction>* v) override
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(MeanFunction));
		}
		void on(std::shared_ptr<DifferentiableFunction>* v) override
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(DifferentiableFunction));
		}
		void on(std::shared_ptr<Inference>* v) override
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(Inference));
		}
		void on(std::shared_ptr<LossFunction>* v) override
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(LossFunction));
		}
		void on(std::shared_ptr<Tokenizer>* v) override
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(Tokenizer));
		}
		void on(std::shared_ptr<CombinationRule>* v) override
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(CombinationRule));
		}
		void on(std::shared_ptr<KernelNormalizer>* v) override
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(KernelNormalizer));
		}
		void on(std::shared_ptr<Transformer>* v) override
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(Transformer));
		}
		void on(std::shared_ptr<MachineEvaluation>* v) override
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(MachineEvaluation));
		}
		void on(std::shared_ptr<StructuredModel>* v) override
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(StructuredModel));
		}
		void on(std::shared_ptr<FactorType>* v) override
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(FactorType));
		}
		void on(std::shared_ptr<ParameterObserver>* v) override
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(ParameterObserver));
		}
		void on(std::shared_ptr<Distribution>* v) override
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(Distribution));
		}
		void on(std::shared_ptr<GaussianProcess>* v) override
		{
			return on_impl(v, SHOGUN_GET_SWIG_TYPE(GaussianProcess));
		}
		void on(std::shared_ptr<Alphabet>* v) override
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
				error("Unexpected error while creating the object");

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
		create_object<SGObject>(name, PT_NOT_GENERIC, visitor);
		return visitor->pyobj();
	}
}
%}
