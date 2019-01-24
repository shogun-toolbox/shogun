#include <gtest/gtest.h>
#include "internal/SVMRegressorConverter.h"

#include "ShogunCoreML.h"

#include <shogun/kernel/LinearKernel.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/kernel/PolyKernel.h>
#include <shogun/kernel/SigmoidKernel.h>

#include <shogun/regression/svr/LibSVR.h>
#include <shogun/regression/svr/SVRLight.h>
#include <shogun/regression/KernelRidgeRegression.h>
#include <shogun/regression/KRRNystrom.h>

#include <shogun/features/DenseFeatures.h>

#include "format/Model.pb.h"
#include "format/SVM.pb.h"

#include "../../tests/unit/environments/LinearTestEnvironment.h"

using namespace shogun;
using namespace shogun::coreml;

extern LinearTestEnvironment* linear_test_env;

template <typename T>
class SVMRegressorTest : public ::testing::Test {};

using SVMRegressorTypes = ::testing::Types<CLibSVR, CSVRLight, CKernelRidgeRegression, CKRRNystrom>;
TYPED_TEST_CASE(SVMRegressorTest, SVMRegressorTypes);

TYPED_TEST(SVMRegressorTest, convert)
{
    auto mock_data = linear_test_env->get_one_dimensional_regression_data(true);
	auto labels_train = (CLabels*) mock_data->get_labels_train();
	auto features_train = mock_data->get_features_train();

	// TODO: make kernel a parameter?
	auto k = some<CLinearKernel>();
	//auto k = some<CGaussianKernel>();
	//auto k = some<CPolyKernel>();
	//auto k = some<CSigmoidKernel>();
	k->init(features_train, features_train);
	auto m = some<TypeParam>();
	m->put("kernel", (CKernel*) k.get());
	m->put("labels", labels_train);
	m->train();

	auto x = m->apply_regression(mock_data->get_features_test());
	mock_data->get_features_test()->get_feature_matrix().display_matrix();
	x->get_labels().display_vector();

    auto converter = std::make_shared<SVMRegressorConverter>(m);
    auto machine_spec = converter->description();
    auto spec = machine_spec->supportvectorregressor();

	// check rho
	ASSERT_EQ(-m->get_bias(), spec.rho());

	// check coeffs
	auto coeffs = m->get_alphas();
	auto coeffs_spec = spec.coefficients();
	ASSERT_EQ(coeffs.size(), coeffs_spec.alpha_size());

	int ctr = 0;
	for (auto c: coeffs)
		ASSERT_EQ(c, coeffs_spec.alpha(ctr++));

	// check kernel
	ASSERT_TRUE(spec.has_kernel());
	auto kernel_spec = spec.kernel();
//	ASSERT_TRUE(kernel_spec.has_rbfkernel());

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

	auto out = convert(m);
    out->save(m->get_name());
}
