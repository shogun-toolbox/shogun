#ifndef __SVM_REGRESSOR_CONVERTER_H__
#define __SVM_REGRESSOR_CONVERTER_H__

#include "internal/SVMConverter.h"

#include <shogun/classifier/svm/SVM.h>

namespace shogun
{
	namespace coreml
	{
		using SVMRegressorConverterType = CoreMLConverter<CSVM, CoreML::Specification::SupportVectorRegressor>;
		class SVMRegressorConverter: public SVMConverter<CSVM, CoreML::Specification::SupportVectorRegressor>
		{
		public:
			explicit SVMRegressorConverter(const CMachine* m);
			virtual ~SVMRegressorConverter() = default;
		};
	}
}

#endif
