#ifndef __SVM_CONVERTER_H__
#define __SVM_CONVERTER_H__

#include "CoreMLConverter.h"
#include "format/FeatureTypes.pb.h"

#include <shogun/classifier/svm/SVM.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/SparseFeatures.h>

namespace shogun
{
	namespace coreml
	{
		template<class I, class O>
		class SVMConverter: public CoreMLConverter<I, O>
		{
		public:
			explicit SVMConverter(const CMachine* m): CoreMLConverter<I, O>(m)
			{
				this->set_model_interface();
			}

		protected:
			::CoreML::Specification::FeatureType* input_feature_type() const override
			{
				auto input_feature_type = new ::CoreML::Specification::FeatureType();
				auto m = static_cast<const I*>(this->m_machine);
				auto kernel = m->get_kernel();
				if (kernel == nullptr)
					throw std::runtime_error("Machine has no kernel set!");

				auto lhs = kernel->get_lhs();
				if (lhs == nullptr)
				{
					SG_UNREF(kernel);
					throw std::runtime_error("Kernel has no left handside features set!");
				}

				input_feature_type->mutable_multiarraytype()->add_shape(input_dimension(lhs));
				input_feature_type->mutable_multiarraytype()->set_datatype(extract_array_type(lhs));

				SG_UNREF(kernel);
				SG_UNREF(lhs);
				return input_feature_type;
			}
		private:
			CoreML::Specification::ArrayFeatureType_ArrayDataType extract_array_type(const CFeatures* f) const
			{
				auto feature_type = f->get_feature_type();
				if (feature_type == F_SHORT || feature_type == F_LONG || feature_type == F_INT
					|| feature_type == F_UINT)
					return CoreML::Specification::ArrayFeatureType_ArrayDataType::ArrayFeatureType_ArrayDataType_INT32;
				else if (feature_type == F_SHORTREAL)
					return CoreML::Specification::ArrayFeatureType_ArrayDataType::ArrayFeatureType_ArrayDataType_FLOAT32;
				else if (feature_type == F_DREAL)
					return CoreML::Specification::ArrayFeatureType_ArrayDataType::ArrayFeatureType_ArrayDataType_DOUBLE;

				throw std::runtime_error("Not specifiable input type");
			}

			index_t input_dimension(const CFeatures* f) const
			{
				switch (f->get_feature_class())
				{
					case C_DENSE:
						return f->as<const CDotFeatures>()->get_dim_feature_space();
					case C_SPARSE:
						return f->as<const CDotFeatures>()->get_dim_feature_space();
					default:
						throw std::runtime_error("Unsupported input type!");
				}
			}
		};

		template<typename T>
		bool set_support_vectors(const CSVM* svm, T* machine_spec)
		{
			auto svs = svm->get_support_vectors();
			auto k = svm->get_kernel();
			auto lhs = k->get_lhs();
			if (lhs == nullptr)
			{
				SG_UNREF(k);
				throw std::runtime_error("Features are not set in kernel (required for support vectors), cannot export to CoreML!");
			}

			bool result = false;
			switch (lhs->get_feature_class())
			{
				case C_DENSE:
				{
					auto svs_spec = machine_spec->mutable_densesupportvectors();
					// FIXME: support all CDenseFeatures type!
					auto dense_features = lhs->as<CDenseFeatures<float64_t>>();
					for (auto sv_idx: svs)
					{
						auto sv = dense_features->get_feature_vector(sv_idx);
						sv.display_vector();
						auto sv_spec = svs_spec->add_vectors();
						for (auto v: sv)
							sv_spec->add_values(v);
					}
					result = true;
					break;
				}
				case C_SPARSE:
				{
					// FIXME: support all CDenseFeatures type!
					auto sparse_features = lhs->as<CSparseFeatures<float64_t>>();
					auto svs_spec = machine_spec->mutable_sparsesupportvectors();
					for (auto sv_idx: svs)
					{
						auto sv = sparse_features->get_sparse_feature_vector(sv_idx);
						auto sv_spec = svs_spec->add_vectors();
						for (index_t i = 0; i < sv.num_feat_entries; ++i)
						{
							auto node = sv_spec->add_nodes();
							node->set_value(sv.features[i].entry);
							node->set_index(sv.features[i].feat_index);
						}
					}
					result = true;
					break;
				}
				default:
					SG_UNREF(lhs);
					SG_UNREF(k);
					throw std::runtime_error("CoreML does not support the provided feature class!");

			}
			SG_UNREF(lhs);
			SG_UNREF(k);

			return false;
		}
	}
}

#endif
