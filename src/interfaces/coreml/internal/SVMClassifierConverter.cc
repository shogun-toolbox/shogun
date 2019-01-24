#include "internal/SVMClassifierConverter.h"
#include "internal/KernelConverter.h"
#include "internal/SVMConverter.h"

#include "format/SVM.pb.h"

#include <shogun/classifier/svm/SVM.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/SparseFeatures.h>

#include <vector>

using namespace CoreML;
using namespace std;

namespace shogun
{
	namespace coreml
	{
		template<>
		const unordered_set<string> SVMClassifierConverterType::supported_types
			= {"SVMLightOneClass", "SVMLight", "MPDSVM", "LibSVMOneClass", "LibSVM", "GNPPSVM"};


		SVMClassifierConverter::SVMClassifierConverter(const CMachine* m):
			SVMConverter<CSVM, CoreML::Specification::SupportVectorClassifier>(m)
		{
			m_spec->set_specificationversion(SPECIFICATION_VERSION);
			m_spec->set_allocated_supportvectorclassifier(CoreMLConverter::convert(static_cast<const input_type*>(m)));
		}

		static void convert_csvm_classifiers(vector<const CSVM*>& ms, CoreML::Specification::SupportVectorClassifier* spec)
		{
			for (int i = 0; i < ms.size(); ++i)
			{
				// num of support vectors per class
				spec->add_numberofsupportvectorsperclass(ms[i]->get_num_support_vectors());

				// set coefficients
				auto coeffs = ms[i]->get_alphas();
				auto coeffs_spec = spec->add_coefficients();
				for (auto c: coeffs)
					coeffs_spec->add_alpha(c);

				// set bias
				spec->add_rho(ms[i]->get_bias());

				// set support vectors
				set_support_vectors(ms[i], spec);
			}

			// set labels
			if (ms.size() == 1)
			{
				spec->mutable_int64classlabels()->add_vector(-1);
				spec->mutable_int64classlabels()->add_vector(1);
			}
			else
			{
				for (int i = 0; i < ms.size(); ++i)
					spec->mutable_int64classlabels()->add_vector(i);
			}
		}

		template<>
		void SVMClassifierConverterType::convert(const CSVM* svm, CoreML::Specification::SupportVectorClassifier* spec)
		{
			vector<const CSVM*> ms {svm};

			// set kernel
			auto kernel = svm->get<CKernel*>("kernel");
			spec->set_allocated_kernel(KernelConverter::convert(kernel));

			convert_csvm_classifiers(ms, spec);
		}

		REGISTER_CONVERTER(SVMClassifierConverter, SVMClassifierConverterType::supported_types)

		template<>
		const unordered_set<string> MulticlassSVMClassifierConverterType::supported_types
			= {"MulticlassLibSVM", "GMNPSVM"};

		MulticlassSVMClassifierConverter::MulticlassSVMClassifierConverter(const CMachine* m):
			SVMConverter<CMulticlassSVM, CoreML::Specification::SupportVectorClassifier>(m)
		{
			m_spec->set_specificationversion(SPECIFICATION_VERSION);
			m_spec->set_allocated_supportvectorclassifier(CoreMLConverter::convert(static_cast<const input_type*>(m)));
		}

		template<>
		void MulticlassSVMClassifierConverterType::convert(const CMulticlassSVM* svm, CoreML::Specification::SupportVectorClassifier* spec)
		{
			vector<const CSVM*> ms;
			auto strategy = svm->get_multiclass_strategy();
			auto num_classes = strategy->get_num_classes();
			SG_UNREF(strategy);

			//get_num_machines()
			for (auto i = 0; i < num_classes; ++i)
				ms.push_back(svm->get_svm(i));

			// set kernel
			auto kernel = svm->get<CKernel*>("kernel");
			spec->set_allocated_kernel(KernelConverter::convert(kernel));

			convert_csvm_classifiers(ms, spec);
		}

		REGISTER_CONVERTER(MulticlassSVMClassifierConverter, MulticlassSVMClassifierConverterType::supported_types)
	}
}
