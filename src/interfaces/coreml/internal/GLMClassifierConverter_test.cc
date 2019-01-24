#include <gtest/gtest.h>
#include "GLMClassifierConverter.h"
#include "ShogunCoreML.h"

#include "format/Model.pb.h"
#include "format/GLMClassifier.pb.h"

#include <shogun/classifier/AveragedPerceptron.h>
#include <shogun/classifier/LDA.h>
#include <shogun/classifier/Perceptron.h>
#include <shogun/classifier/svm/LibLinear.h>
#include <shogun/classifier/svm/NewtonSVM.h>
#include <shogun/classifier/svm/SGDQN.h>
#include <shogun/classifier/svm/SVMOcas.h>
#include <shogun/multiclass/MulticlassLibLinear.h>
#include <shogun/multiclass/MulticlassLogisticRegression.h>
#include <shogun/multiclass/MulticlassOCAS.h>

#include "../../tests/unit/environments/LinearTestEnvironment.h"
#include "../../tests/unit/environments/MultiLabelTestEnvironment.h"

using namespace shogun;
using namespace shogun::coreml;

extern LinearTestEnvironment* linear_test_env;
extern MultiLabelTestEnvironment* multilabel_test_env;

template <typename T>
class GLMClassifier : public ::testing::Test {};

using GLMClassifierTypes = ::testing::Types<CAveragedPerceptron, CLDA, CPerceptron, CNewtonSVM, CLibLinear, CSGDQN, CSVMOcas>;
TYPED_TEST_CASE(GLMClassifier, GLMClassifierTypes);

TYPED_TEST(GLMClassifier, convert)
{
    auto mockData = linear_test_env->getBinaryLabelData();
	auto train_feats = mockData->get_features_train();
	auto train_labels = mockData->get_labels_train();

    auto m = some<TypeParam>();
    m->put("labels", train_labels);
	m->train(train_feats);

    auto x = m->apply_binary(mockData->get_features_test());
    mockData->get_features_test()->get_feature_matrix().display_matrix();
    x->get_labels().display_vector();

    auto converter = std::make_shared<GLMClassifierConverter>(m);
    auto machine_spec = converter->description();
    auto spec = machine_spec->glmclassifier();

    ASSERT_EQ(1, spec.weights_size());
    auto w = m->get_w();
    auto w_spec = spec.weights(0);
    ASSERT_EQ(w.vlen, w_spec.value_size());
    int ctr = 0;
    for (auto v: w)
        ASSERT_EQ(v, w_spec.value(ctr++));

    ASSERT_EQ(1, spec.offset_size());
    ASSERT_EQ(m->get_bias(), spec.offset(0));

    auto out = convert(m);
    out->save(m->get_name());
}

template <typename T>
class GLMClassifierMulticlass : public ::testing::Test {};

using GLMClassifierMulticlassTypes = ::testing::Types <CMulticlassLibLinear, CMulticlassOCAS, CMulticlassLogisticRegression>;
TYPED_TEST_CASE(GLMClassifierMulticlass, GLMClassifierMulticlassTypes);

TYPED_TEST(GLMClassifierMulticlass, convert)
{
    auto mockData = multilabel_test_env->getMulticlassFixture();

    auto train_feats = mockData->get_features_train();
    auto train_labels = mockData->get_labels_train();

    auto m = some<TypeParam>();
    m->put("labels", train_labels);
    m->train(train_feats);

    auto x = m->apply_multiclass(mockData->get_features_test());
    mockData->get_features_test()->get_feature_matrix().display_matrix();
    x->get_labels().display_vector();

    auto converter = std::make_shared<MulticlassGLMClassifierConverter>(m);
    auto machine_spec = converter->description();

    auto out = convert(m);
    out->save(m->get_name());

//        ASSERT_EQ(m->, spec->weights_size());
}
