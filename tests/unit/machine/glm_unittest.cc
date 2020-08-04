/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Both the documentation and the code is heavily inspired by pyGLMnet.:
 * https://github.com/glm-tools/pyglmnet/
 *
 * Author: Tej Sukhatme
 */

#include <gtest/gtest.h>
#include <random>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/machine/GLM.h>
#include <shogun/machine/LinearMachine.h>
#include <shogun/mathematics/RandomNamespace.h>
#include <shogun/mathematics/UniformRealDistribution.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/preprocessor/NormOne.h>
#include <shogun/preprocessor/PruneVarSubMean.h>

using namespace shogun;

// Generate the Data that N is greater than D
std::tuple<SGMatrix<float64_t>, SGVector<float64_t>> generate_train_data()
{
	SGVector<float64_t> labels = SGVector<float64_t>({11, 4, 9, 7, 8});
	SGMatrix<float64_t> features = SGMatrix<float64_t>({
	    {0.40015721, 0.97873798, 2.2408932},
	    {1.86755799, -0.97727788, 0.95008842},
	    {-0.15135721, -0.10321885, 0.4105985},
	    {0.14404357, 1.45427351, 0.76103773},
	    {0.12167502, 0.44386323, 0.33367433},
	});

	return {features, labels};
}

std::tuple<SGMatrix<float64_t>, SGVector<float64_t>> generate_test_data()
{
	SGVector<float64_t> labels = SGVector<float64_t>({2, 3, 7, 7, 4});
	SGMatrix<float64_t> features = SGMatrix<float64_t>({
	    {1.49407907, -0.20515826, 0.3130677},
	    {-0.85409574, -2.55298982, 0.6536186},
	    {0.8644362, -0.74216502, 2.26975462},
	    {-1.45436567, 0.04575852, -0.18718385},
	    {1.53277921, 1.46935877, 0.15494743},
	});

	return {features, labels};
}

TEST(GLM, GLM_basic_test)
{
	const auto& [Xtrain, ytrain] = generate_train_data();
	const auto& [Xtest, ytest] = generate_test_data();

	auto features_train = std::make_shared<DenseFeatures<float64_t>>(Xtrain);
	auto labels_train = std::make_shared<RegressionLabels>(ytrain);

	auto features_test = std::make_shared<DenseFeatures<float64_t>>(Xtest);

	auto glm = std::make_shared<GLM>(POISSON, 0.5, 0.1, 2e-1, 1000, 1e-6, 2.0);

	glm->set_bias(0.44101309);
	glm->set_w(SGVector<float64_t>({0.1000393, 0.2446845, 0.5602233}));

	glm->set_labels(labels_train);

	glm->train(features_train);

	auto labels_predict = glm->apply_regression(features_test);

	/** Labels calculated here:
	 * https://gist.github.com/Hephaestus12/8f303604045308202ae06b4845cf315c */

	SGVector<float64_t> labels_pyglmnet(
	    {1.89767309, 9.21466271, 3.21613202, 11.89327037, 1.83352229});

	float64_t epsilon = 1e-7;

	for (index_t i = 0; i < labels_predict->get_num_labels(); ++i)
		EXPECT_NEAR(labels_predict->get_label(i), labels_pyglmnet[i], epsilon);
}

TEST(GLMCostFunction, GLM_POISSON_gradient_test)
{
	const auto& [Xtrain, ytrain] = generate_train_data();

	auto glm_cost = std::make_shared<GLMCostFunction>();

	float64_t bias = -0.347144;
	SGVector<float64_t> w =
	    SGVector<float64_t>({0.295102, 0.423805, -0.125878});

	SGVector<float64_t> grad_w = glm_cost->get_gradient_weights(
	    Xtrain, ytrain, w, bias, 0.1, 0.5, true, 2.0, POISSON);
	float64_t grad_bias = glm_cost->get_gradient_bias(
	    Xtrain, ytrain, w, bias, true, 2.0, POISSON);

	/** gradient calculated here:
	 * https://gist.github.com/Hephaestus12/84ace46a18deed6157dca0a3e3640bfe */

	SGVector<float64_t> pyglmnet_grad_w =
	    SGVector<float64_t>({-2.10448752, -3.4498384, -7.19517773});
	float64_t pyglmnet_grad_bias = -6.933939890424355;
	float64_t epsilon = 1e-7;

	for (auto i : range(grad_w.vlen))
		EXPECT_NEAR(grad_w[i], pyglmnet_grad_w[i], epsilon);

	EXPECT_NEAR(grad_bias, pyglmnet_grad_bias, epsilon);
}
