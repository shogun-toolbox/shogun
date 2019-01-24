#include <gtest/gtest.h>
#include "internal/GLMRegressorConverter.h"

#include "ShogunCoreML.h"

#include "format/GLMRegressor.pb.h"
#include "format/Model.pb.h"

#include <shogun/regression/svr/LibLinearRegression.h>
#include <shogun/regression/LeastAngleRegression.h>
#include <shogun/regression/LeastSquaresRegression.h>
#include <shogun/regression/LinearRidgeRegression.h>

#include "../../tests/unit/environments/LinearTestEnvironment.h"

using namespace shogun;
using namespace shogun::coreml;

extern LinearTestEnvironment* linear_test_env;

template <typename T>
class GLMRegressorTest : public ::testing::Test {};

using GLMRegressorTypes = ::testing::Types<CLibLinearRegression, CLeastAngleRegression, CLeastSquaresRegression, CLinearRidgeRegression>;
TYPED_TEST_CASE(GLMRegressorTest, GLMRegressorTypes);

TYPED_TEST(GLMRegressorTest, convert)
{
	auto mock_data = linear_test_env->get_one_dimensional_regression_data(true);
	auto labels_train = (CLabels*) mock_data->get_labels_train();
	auto features_train = mock_data->get_features_train();

	auto m = some<TypeParam>();
	m->put("labels", labels_train);
	m->train(features_train);

	auto x = m->apply_regression(mock_data->get_features_test());
	mock_data->get_features_test()->get_feature_matrix().display_matrix();
	x->get_labels().display_vector();

	auto converter = std::make_shared<GLMRegressorConverter>(m);
	auto machine_spec = converter->description();
	auto spec = machine_spec->glmregressor();

	ASSERT_EQ(1, spec.weights_size());
	auto w = m->get_w();
	auto w_spec = spec.weights(0);
	ASSERT_EQ(w.vlen, w_spec.value_size());
	int ctr = 0;
	for (auto v: w)
		ASSERT_EQ(v, w_spec.value(ctr++));

	ASSERT_EQ(1, spec.offset_size());
	ASSERT_EQ(m->get_bias(), spec.offset(0));

	ASSERT_EQ(
		CoreML::Specification::GLMRegressor_PostEvaluationTransform::GLMRegressor_PostEvaluationTransform_NoTransform,
		spec.postevaluationtransform());

	auto out = convert(m);
    out->save(m->get_name());
}
