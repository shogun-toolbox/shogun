#include <gtest/gtest.h>
#include "internal/SVMClassifierConverter.h"
#include "ShogunCoreML.h"

#include "format/SVM.pb.h"

#include <shogun/kernel/GaussianKernel.h>

#include <shogun/classifier/svm/LibSVM.h>
#include <shogun/classifier/svm/LibSVMOneClass.h>
#include <shogun/classifier/svm/SVMLight.h>
#include <shogun/classifier/svm/MPDSVM.h>
#include <shogun/classifier/svm/GNPPSVM.h>
#include <shogun/multiclass/GMNPSVM.h>
#include <shogun/multiclass/MulticlassLibSVM.h>

#include "../../tests/unit/environments/LinearTestEnvironment.h"
#include "../../tests/unit/environments/MultiLabelTestEnvironment.h"

using namespace shogun;
using namespace shogun::coreml;

extern LinearTestEnvironment* linear_test_env;
extern MultiLabelTestEnvironment* multilabel_test_env;

template <typename T>
class SVMClassifier : public ::testing::Test {};
using SVMClassifierTypes = ::testing::Types<CLibSVM, CSVMLight, CMPDSVM, CLibSVMOneClass, CLibSVM, CGNPPSVM>;
TYPED_TEST_CASE(SVMClassifier, SVMClassifierTypes);

TYPED_TEST(SVMClassifier, convert)
{
    auto mockData = linear_test_env->getBinaryLabelData();
	auto train_feats = mockData->get_features_train();
	auto train_labels = mockData->get_labels_train();

	// TODO: make kernel a parameter?
	auto k = some<CGaussianKernel>(10);
	k->init(train_feats, train_feats);
	auto m = some<TypeParam>();
	m->put("kernel", (CKernel*) k.get());
	m->put("labels", train_labels);
	m->train();

    auto x = m->apply_binary(mockData->get_features_test());
    mockData->get_features_test()->get_feature_matrix().display_matrix();
    x->get_labels().display_vector();

    auto converter = std::make_shared<SVMClassifierConverter>(m);
    auto machine_spec = converter->description();
    auto spec = machine_spec->supportvectorclassifier();

    // check kernel
	ASSERT_TRUE(spec.has_kernel());
	auto kernel_spec = spec.kernel();
	ASSERT_TRUE(kernel_spec.has_rbfkernel());

	// check rho
	auto rhos = spec.rho();
	ASSERT_EQ(1, rhos.size());
	ASSERT_EQ(m->get_bias(), rhos[0]);

	// check coeffs
	auto coeffs = m->get_alphas();
	auto coeffs_spec = spec.coefficients();
	ASSERT_EQ(1, coeffs_spec.size());
	ASSERT_EQ(coeffs.size(), coeffs_spec[0].alpha_size());

	int ctr = 0;
	for (auto c: coeffs)
		ASSERT_EQ(c, coeffs_spec[0].alpha(ctr++));
/*
	// check support vectors
	ASSERT_TRUE(spec.has_densesupportvectors());
	auto svs_idx = m->get_support_vectors();
	auto svs_spec = spec.densesupportvectors();

	ASSERT_EQ(svs_idx.vlen, svs_spec.vectors_size());
	ctr = 0;
	for (auto idx: svs_idx)
	{
		auto sv_spec = svs_spec.vectors(ctr++);
		auto sv = features_train->get_feature_vector(idx);
		ASSERT_EQ(sv.vlen, sv_spec.values_size());

		int j = 0;
		for (auto v: sv)
			ASSERT_EQ(v, sv_spec.values(j++));
	}

*/
    auto out = convert(m);
    out->save(m->get_name());
}

template <typename T>
class SVMClassifierMulticlass : public ::testing::Test {};
using SVMClassifierMulticlassTypes = ::testing::Types<CMulticlassLibSVM, CGMNPSVM>;
TYPED_TEST_CASE(SVMClassifierMulticlass, SVMClassifierMulticlassTypes);
TYPED_TEST(SVMClassifierMulticlass, convert)
{
    auto mockData = multilabel_test_env->getMulticlassFixture();

    auto train_feats = mockData->get_features_train();
    auto train_labels = mockData->get_labels_train();

	// TODO: make kernel a parameter?
	auto k = some<CGaussianKernel>(10);
	k->init(train_feats, train_feats);
	auto m = some<TypeParam>();
	m->put("kernel", (CKernel*) k.get());
	m->put("labels", train_labels);
	m->put("C", 1.0);
	m->train();

    auto x = m->apply_multiclass(mockData->get_features_test());
    mockData->get_features_test()->get_feature_matrix().display_matrix();
    x->get_labels().display_vector();

    //auto converter = std::make_shared<GLMClassifierConverter>(m);
    //auto machine_spec = converter->description();
    auto out = convert(m);
    out->save(m->get_name());

//        ASSERT_EQ(m->, spec->weights_size());
}