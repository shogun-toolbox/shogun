#ifndef __GLM_CLASSIFIER_CONVERTER_H__
#define __GLM_CLASSIFIER_CONVERTER_H__

#include "CoreMLConverter.h"

#include <shogun/machine/LinearMachine.h>
#include <shogun/machine/LinearMulticlassMachine.h>

namespace shogun
{
	namespace coreml
	{
		using GLMClassifierConverterType = CoreMLConverter<CLinearMachine, CoreML::Specification::GLMClassifier>;
		class GLMClassifierConverter: public GLMClassifierConverterType
		{
		public:
			explicit GLMClassifierConverter(const CMachine* m);
			virtual ~GLMClassifierConverter() = default;
		protected:
			::CoreML::Specification::FeatureType* input_feature_type() const override;
		};

		using MulticlassGLMClassifierConverterType = CoreMLConverter<CLinearMulticlassMachine, CoreML::Specification::GLMClassifier>;
		class MulticlassGLMClassifierConverter: public MulticlassGLMClassifierConverterType
		{
		public:
			explicit MulticlassGLMClassifierConverter(const CMachine* m);
			virtual ~MulticlassGLMClassifierConverter() = default;
		protected:
			::CoreML::Specification::FeatureType* input_feature_type() const override;
		};
	}
}

#endif
