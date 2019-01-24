#ifndef __GLM_REGRESSOR_CONVERTER_H__
#define __GLM_REGRESSOR_CONVERTER_H__

#include "CoreMLConverter.h"

#include <shogun/machine/LinearMachine.h>

namespace shogun
{
	namespace coreml
	{
		using GLMRegressorConverterType = CoreMLConverter<CLinearMachine, CoreML::Specification::GLMRegressor>;
		class GLMRegressorConverter: public GLMRegressorConverterType
		{
		public:
			explicit GLMRegressorConverter(const CMachine* m);
			virtual ~GLMRegressorConverter() = default;
		protected:
			//::CoreML::Specification::FeatureType* input_feature_type() const override;
			virtual ::CoreML::Specification::FeatureType* input_feature_type() const override;
		};
	}
}

#endif
