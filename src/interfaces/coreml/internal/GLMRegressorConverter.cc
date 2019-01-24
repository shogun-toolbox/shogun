#include "GLMRegressorConverter.h"

#include "format/GLMRegressor.pb.h"

using namespace CoreML;
using namespace std;

namespace shogun
{
	namespace coreml
	{
		template<>
		const unordered_set<string> GLMRegressorConverterType::supported_types = {
			"LibLinearRegression",
			"LeastAngleRegression",
			"LeastSquaresRegression",
			"LinearRidgeRegression"};

		GLMRegressorConverter::GLMRegressorConverter(const CMachine* m):
			GLMRegressorConverterType(m)
		{
			set_model_interface();

			m_spec->set_specificationversion(SPECIFICATION_VERSION);
			m_spec->set_allocated_glmregressor(
				CoreMLConverter::convert(static_cast<const input_type*>(m)));
		}

		template<>
		void GLMRegressorConverterType::convert(
			const CLinearMachine* lr,
			Specification::GLMRegressor* spec)
		{
			REQUIRE(lr != NULL, "No machine has been provided")
			REQUIRE(spec != NULL, "No CoreML specification has been provided")

			// set weights
			auto w = lr->get_w();
			auto w_spec = spec->add_weights();
			for (auto v: w)
				w_spec->add_value(v);

			// set offset
			spec->add_offset(lr->get_bias());

			// set post evalution transform
			spec->set_postevaluationtransform(Specification::GLMRegressor::NoTransform);
			// other possible values:
			// Specification::GLMRegressor::Logit
			// Specification::GLMRegressor::Probit
		}

		Specification::FeatureType* GLMRegressorConverter::input_feature_type() const
		{
			auto input_feature_type = new Specification::FeatureType();
			input_feature_type->mutable_multiarraytype()
				->add_shape(static_cast<const input_type*>(m_machine)->get_w().vlen);
			input_feature_type->mutable_multiarraytype()
				->set_datatype(Specification::ArrayFeatureType_ArrayDataType::ArrayFeatureType_ArrayDataType_DOUBLE);
			return input_feature_type;
		}

		REGISTER_CONVERTER(GLMRegressorConverter, GLMRegressorConverterType::supported_types)
	}
}
