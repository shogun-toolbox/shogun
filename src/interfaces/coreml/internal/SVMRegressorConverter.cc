#include "internal/SVMRegressorConverter.h"
#include "internal/KernelConverter.h"

#include "format/SVM.pb.h"

#include <shogun/classifier/svm/SVM.h>

#include <iostream>

using namespace shogun;
using namespace shogun::coreml;
using namespace CoreML;
namespace shogun
{
	namespace coreml
	{

		template<>
		const std::unordered_set<std::string> SVMRegressorConverterType::supported_types
			= {"LibSVR", "SVRLight", "KernelRidgeRegression", "KRRNystrom"};

		SVMRegressorConverter::SVMRegressorConverter(const CMachine* m):
			SVMConverter<CSVM, CoreML::Specification::SupportVectorRegressor>(m)
		{
			m_spec->set_specificationversion(SPECIFICATION_VERSION);
			m_spec->set_allocated_supportvectorregressor(CoreMLConverter::convert(static_cast<const input_type*>(m)));
		}

		template<>
		void SVMRegressorConverterType::convert(const CSVM* svm, CoreML::Specification::SupportVectorRegressor* spec)
		{
			// set kernel
			auto kernel = svm->get<CKernel*>("kernel");
			spec->set_allocated_kernel(KernelConverter::convert(kernel));

			// set coefficients
			auto coeffs = svm->get_alphas();
			auto coeffs_spec = spec->mutable_coefficients();
			for (auto c: coeffs)
				coeffs_spec->add_alpha(c);

			// set rho
			spec->set_rho(-svm->get_bias());

			// set support vectors
			set_support_vectors(svm, spec);
		}

		REGISTER_CONVERTER(SVMRegressorConverter, SVMRegressorConverterType::supported_types)
	}
}
