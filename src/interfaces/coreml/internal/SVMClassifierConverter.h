#ifndef __SVM_CLASSIFIER_CONVERTER_H__
#define __SVM_CLASSIFIER_CONVERTER_H__

#include "internal/SVMConverter.h"

#include <shogun/classifier/svm/SVM.h>
#include <shogun/multiclass/MulticlassSVM.h>

namespace shogun
{
	namespace coreml
	{
		using SVMClassifierConverterType = CoreMLConverter<CSVM, CoreML::Specification::SupportVectorClassifier>;
		class SVMClassifierConverter: public SVMConverter<CSVM, CoreML::Specification::SupportVectorClassifier>
		{
		public:
			explicit SVMClassifierConverter(const CMachine* m);
			virtual ~SVMClassifierConverter() = default;
		};

		using MulticlassSVMClassifierConverterType = CoreMLConverter<CMulticlassSVM, CoreML::Specification::SupportVectorClassifier>;
		class MulticlassSVMClassifierConverter: public SVMConverter<CMulticlassSVM, CoreML::Specification::SupportVectorClassifier>
		{
		public:
			explicit MulticlassSVMClassifierConverter(const CMachine* m);
			virtual ~MulticlassSVMClassifierConverter() = default;
		};

	}
}

#endif
