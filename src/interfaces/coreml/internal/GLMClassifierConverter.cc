#include "GLMClassifierConverter.h"

#include "format/FeatureTypes.pb.h"
#include "format/GLMClassifier.pb.h"

#include <shogun/machine/LinearMulticlassMachine.h>

using namespace CoreML;

namespace shogun
{
	namespace coreml
	{
		template<>
		const std::unordered_set<std::string> GLMClassifierConverterType::supported_types
			= {"SVMOcas", "AveragedPerceptron", "LDA", "Perceptron", "NewtonSVM", "LibLinear", "SGDQN"};

		GLMClassifierConverter::GLMClassifierConverter(const CMachine* m): GLMClassifierConverterType(m)
		{
			m_spec->set_specificationversion(SPECIFICATION_VERSION);
			m_spec->set_allocated_glmclassifier(CoreMLConverter::convert(static_cast<const input_type*>(m)));
			set_model_interface();
		}

		template<>
		void CoreMLConverter<CLinearMachine, CoreML::Specification::GLMClassifier>::convert(const CLinearMachine* lm, Specification::GLMClassifier* spec)
		{
			REQUIRE(lm != NULL, "No machine has been provided")
			REQUIRE(spec != NULL, "No CoreML specification has been provided")

			// set weights
			auto w = lm->get_w();
			auto w_spec = spec->add_weights();
			for (auto v: w)
				w_spec->add_value(v);

			// set offset
			spec->add_offset(lm->get_bias());

			// FIXME: set post evalution transform
			spec->set_postevaluationtransform(Specification::GLMClassifier::Logit);

			// set labels
			spec->mutable_int64classlabels()->add_vector(-1);
			spec->mutable_int64classlabels()->add_vector(1);

			// encoding
			spec->set_classencoding(Specification::GLMClassifier_ClassEncoding::GLMClassifier_ClassEncoding_ReferenceClass);
		}

		Specification::FeatureType* GLMClassifierConverter::input_feature_type() const
		{
			auto input_feature_type = new Specification::FeatureType();
			input_feature_type->mutable_multiarraytype()->add_shape(static_cast<const input_type*>(m_machine)->get_w().vlen);
			input_feature_type->mutable_multiarraytype()->set_datatype(Specification::ArrayFeatureType_ArrayDataType::ArrayFeatureType_ArrayDataType_DOUBLE);
			return input_feature_type;
		}

		template<>
		const std::unordered_set<std::string> MulticlassGLMClassifierConverterType::supported_types
			= {"MulticlassLibLinear", "MulticlassOCAS", "MulticlassLogisticRegression"};

		MulticlassGLMClassifierConverter::MulticlassGLMClassifierConverter(const CMachine* m): MulticlassGLMClassifierConverterType(m)
		{
			m_spec->set_specificationversion(SPECIFICATION_VERSION);
			m_spec->set_allocated_glmclassifier(CoreMLConverter::convert(static_cast<const input_type*>(m)));
			set_model_interface();
		}

		template<>
		void MulticlassGLMClassifierConverterType::convert(const CLinearMulticlassMachine* mc, Specification::GLMClassifier* spec)
		{
			auto strategy = mc->get_multiclass_strategy();
			if (std::string(strategy->get_name()) != "MulticlassOneVsRestStrategy")
				throw std::runtime_error("Unsupported multiclass strategy!");
			else
				spec->set_classencoding(CoreML::Specification::GLMClassifier_ClassEncoding::GLMClassifier_ClassEncoding_OneVsRest);

			auto num_classes = strategy->get_num_classes();
			SG_UNREF(strategy);
			for (auto i = 0; i < num_classes; ++i)
			{
				// set labels
				spec->mutable_int64classlabels()->add_vector(i);

				// set weights
				auto cur_machine = mc->get_machine(i)->as<const CLinearMachine>();
				auto w = cur_machine->get_w();
				auto w_spec = spec->add_weights();
				for (auto v: w)
					w_spec->add_value(v);

				// set offset
				spec->add_offset(cur_machine->get_bias());
			}

			// FIXME: set post evalution transform
			spec->set_postevaluationtransform(Specification::GLMClassifier::Logit);
		}

		Specification::FeatureType* MulticlassGLMClassifierConverter::input_feature_type() const
		{
			auto input_feature_type = new Specification::FeatureType();
			auto f = static_cast<const input_type*>(m_machine)->get_features();
			input_feature_type->mutable_multiarraytype()->add_shape(f->get_dim_feature_space());
			input_feature_type->mutable_multiarraytype()->set_datatype(Specification::ArrayFeatureType_ArrayDataType::ArrayFeatureType_ArrayDataType_DOUBLE);
			SG_UNREF(f);
			return input_feature_type;
		}

		REGISTER_CONVERTER(GLMClassifierConverter, GLMClassifierConverterType::supported_types)
		REGISTER_CONVERTER(MulticlassGLMClassifierConverter, MulticlassGLMClassifierConverterType::supported_types)
	}
}

