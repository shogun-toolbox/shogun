/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 * 
 * Both the documentation and the code is heavily inspired by pyGLMnet.: https://github.com/glm-tools/pyglmnet/
 *
 * Author: Tej Sukhatme
 */

#include <gtest/gtest.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/machine/GLM.h>
#include <shogun/machine/LinearMachine.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/preprocessor/PruneVarSubMean.h>
#include <shogun/preprocessor/NormOne.h>
#include <shogun/mathematics/UniformRealDistribution.h>
#include <shogun/mathematics/RandomNamespace.h>

#include <random>

using namespace shogun;

//Generate the Data that N is greater than D
void generate_train_data(SGMatrix<float64_t> &features, SGVector<float64_t> &labels)
{

	labels = SGVector<float64_t>({7.23514031, 7.23514031, 7.23514031, 7.23514031, 7.23514031});
	features = SGMatrix<float64_t>({{ 0.71307143, -0.67054885, -0.24406853},
								{-0.79774475, -1.65627891,  0.95675428},
								{ 0.96709333,  1.81672959,  0.20911922},
								{ 2.1912712,   0.23820139,  1.07501177},
								{-0.58427793, -0.61855905,  1.27687684}});
}

void generate_test_data(SGMatrix<float64_t> &features, SGVector<float64_t> &labels)
{

	labels = SGVector<float64_t>({7.23514031, 7.23514031, 7.23514031, 7.23514031, 7.23514031});
	features = SGMatrix<float64_t>({{ 1.26465769,  0.05451801, -0.21206714},
								{-0.3447881,  -0.81339926,  1.636931  },
								{ 0.3967461,  -1.6470009,   0.89995864},
								{ 0.65379594,  1.08610417, -0.04911578},
								{ 0.6573247,  -0.1306287,  -0.64715244}});
}

TEST(GLM, GLM_basic_test)
{
	SGMatrix<float64_t> Xtrain(3,5);
	SGVector<float64_t> ytrain(5);
	generate_train_data(Xtrain, ytrain);
	Xtrain.display_matrix("Xtrain");
	ytrain.display_vector("ytrain");

	SGMatrix<float64_t> Xtest(3,5);
	SGVector<float64_t> ytest(5);
	generate_test_data(Xtest, ytest);
	Xtest.display_matrix("Xtest");
	ytest.display_vector("ytrain");

	auto features_train = std::make_shared<DenseFeatures<float64_t> >(Xtrain);
	auto labels_train = std::make_shared<RegressionLabels>(ytrain);

	auto features_test = std::make_shared<DenseFeatures<float64_t> >(Xtest);

	std::cout<<"Making GLM instance.\n";
	auto glm=std::make_shared<GLM>(POISSON, 0.5, 0.1, 2e-1, 1000, 1e-6, 2.0);

	glm->set_bias(0.44101309);
	glm->set_w(SGVector<float64_t>({0.1000393, 0.2446845, 0.5602233}));

	std::cout<<"bais:\t"<<glm->get_bias()<<'\n';
	glm->get_w().display_vector("Weights");

	std::cout<<"Setting labels.\n";
	std::cout<<"Size of labels is: "<<labels_train->get_labels().vlen<<'\n';
	glm->set_labels(labels_train);

	std::cout<<"Training GLM.\n";
	glm->train(features_train);

	std::cout<<"bais:\t"<<glm->get_bias();
	glm->get_w().display_vector("Weights");

	std::cout<<"Applying GLM.\n";
	auto labels_predict = glm->apply_regression(features_test);

	SGVector<float64_t> labels_pyglmnet({5.47606254, 5.62436215, 5.50835709, 5.75757967, 6.01670904});

	float64_t epsilon=0.000001;
	std::cout<<"Comparing results.";
	for ( index_t i = 0; i < labels_predict->get_num_labels(); ++i )
		EXPECT_NEAR(labels_predict->get_label(i), labels_pyglmnet[i], epsilon);
}