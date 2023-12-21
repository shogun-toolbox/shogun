/*
 * Copyright (c) 2014, Shogun Toolbox Foundation
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:

 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Written (W) 2014 Khaled Nasr
 */

#include <gtest/gtest.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/neuralnets/NeuralNetwork.h>
#include <shogun/neuralnets/NeuralInputLayer.h>
#include <shogun/neuralnets/NeuralLogisticLayer.h>
#include <shogun/neuralnets/NeuralSoftmaxLayer.h>
#include <shogun/neuralnets/NeuralRectifiedLinearLayer.h>
#include <shogun/neuralnets/NeuralConvolutionalLayer.h>
#include <shogun/neuralnets/NeuralLayers.h>

#include <vector>

using namespace shogun;

/** Tests gradients computed using backpropagation against gradients computed
 * by numerical approximation. Uses a NeuralLinearLayer-based network.
 */
TEST(NeuralNetwork, backpropagation_linear)
{
	int32_t seed = 10;
	float64_t tolerance = 1e-9;

	std::vector<std::shared_ptr<NeuralLayer>> layers;
	layers.push_back(std::make_shared<NeuralInputLayer>(5));
	layers.push_back(std::make_shared<NeuralInputLayer>(7));
	layers.push_back(std::make_shared<NeuralLinearLayer>(3));
	layers.push_back(std::make_shared<NeuralLinearLayer>(6));
	layers.push_back(std::make_shared<NeuralLinearLayer>(5));
	layers.push_back(std::make_shared<NeuralLinearLayer>(4));
	auto network = std::make_shared<NeuralNetwork>(layers);
	network->put("seed", seed);

	network->connect(0,2);
	network->connect(1,2);
	network->connect(2,3);
	network->connect(2,4);
	network->connect(3,5);
	network->connect(4,5);

	network->initialize_neural_network();
	network->set_l2_coefficient(0.01);
	network->set_l1_coefficient(0.03);

	EXPECT_NEAR(network->check_gradients(), 0.0, tolerance);
}

/** Tests neural layers builder
 */
TEST(NeuralNetwork, neural_layers_builder)
{
	int32_t seed = 10;
	float64_t tolerance = 1e-9;

	auto layers = std::make_shared<NeuralLayers>();
	layers->input(5)
	      ->input(7)
	      ->linear(3)
	      ->linear(6)
	      ->linear(5)
	      ->linear(4);
	auto network = std::make_shared<NeuralNetwork>(layers->done());
	network->put("seed", seed);

	network->connect(0,2);
	network->connect(1,2);
	network->connect(2,3);
	network->connect(2,4);
	network->connect(3,5);
	network->connect(4,5);

	network->initialize_neural_network();
	network->set_l2_coefficient(0.01);
	network->set_l1_coefficient(0.03);

	EXPECT_NEAR(network->check_gradients(), 0.0, tolerance);
}


/** Tests gradients computed using backpropagation against gradients computed
 * by numerical approximation. Uses a NeuralLogisticLayer-based network.
 */
TEST(NeuralNetwork, backpropagation_logistic)
{
	int32_t seed = 10;
	float64_t tolerance = 1e-9;

	std::vector<std::shared_ptr<NeuralLayer>> layers;
	layers.push_back(std::make_shared<NeuralInputLayer>(5));
	layers.push_back(std::make_shared<NeuralInputLayer>(7));
	layers.push_back(std::make_shared<NeuralLogisticLayer>(3));
	layers.push_back(std::make_shared<NeuralLogisticLayer>(6));
	layers.push_back(std::make_shared<NeuralLogisticLayer>(5));
	layers.push_back(std::make_shared<NeuralLogisticLayer>(4));
	auto network = std::make_shared<NeuralNetwork>(layers);
	network->put("seed", seed);

	network->connect(0,2);
	network->connect(1,2);
	network->connect(2,3);
	network->connect(2,4);
	network->connect(3,5);
	network->connect(4,5);

	network->initialize_neural_network();
	network->set_l1_coefficient(0.03);
	network->set_l2_coefficient(0.01);
	EXPECT_NEAR(network->check_gradients(), 0.0, tolerance);

}

/** Tests gradients computed using backpropagation against gradients computed
 * by numerical approximation. Uses a NeuralSoftmaxLayer-based network.
 */
TEST(NeuralNetwork, backpropagation_softmax)
{
	int32_t seed = 10;
	float64_t tolerance = 1e-9;

	std::vector<std::shared_ptr<NeuralLayer>> layers;
	layers.push_back(std::make_shared<NeuralInputLayer>(5));
	layers.push_back(std::make_shared<NeuralInputLayer>(7));
	layers.push_back(std::make_shared<NeuralLinearLayer>(3));
	layers.push_back(std::make_shared<NeuralLinearLayer>(6));
	layers.push_back(std::make_shared<NeuralLinearLayer>(5));
	layers.push_back(std::make_shared<NeuralSoftmaxLayer>(4));
	auto network = std::make_shared<NeuralNetwork>(layers);
	network->put("seed", seed);

	network->connect(0,2);
	network->connect(1,2);
	network->connect(2,3);
	network->connect(2,4);
	network->connect(3,5);
	network->connect(4,5);

	network->initialize_neural_network();
	network->set_l1_coefficient(0.03);
	network->set_l2_coefficient(0.01);
	EXPECT_NEAR(network->check_gradients(), 0.0, tolerance);

}

/** Tests gradients computed using backpropagation against gradients computed
 * by numerical approximation. Uses a NeuralRectifiedLinearLayer-based network.
 */
TEST(NeuralNetwork, backpropagation_rectified_linear)
{
	int32_t seed = 10;
	float64_t tolerance = 1e-9;

	std::vector<std::shared_ptr<NeuralLayer>> layers;
	layers.push_back(std::make_shared<NeuralInputLayer>(5));
	layers.push_back(std::make_shared<NeuralInputLayer>(7));
	layers.push_back(std::make_shared<NeuralRectifiedLinearLayer>(3));
	layers.push_back(std::make_shared<NeuralRectifiedLinearLayer>(6));
	layers.push_back(std::make_shared<NeuralRectifiedLinearLayer>(5));
	layers.push_back(std::make_shared<NeuralLinearLayer>(4));
	auto network = std::make_shared<NeuralNetwork>(layers);
	network->put("seed", seed);

	network->connect(0,2);
	network->connect(1,2);
	network->connect(2,3);
	network->connect(2,4);
	network->connect(3,5);
	network->connect(4,5);

	network->initialize_neural_network();
	network->set_l1_coefficient(0.03);
	network->set_l2_coefficient(0.01);
	EXPECT_NEAR(network->check_gradients(), 0.0, tolerance);

}

/** Tests gradients computed using backpropagation against gradients computed
 * by numerical approximation. Uses a NeuralConvolutionalLayer-based network.
 */
TEST(NeuralNetwork, backpropagation_convolutional)
{
	int32_t seed = 10;
	float64_t tolerance = 1e-9;

	std::vector<std::shared_ptr<NeuralLayer>> layers;
	layers.push_back(std::make_shared<NeuralInputLayer>(6,4));
	layers.push_back(std::make_shared<NeuralInputLayer>(6,4));
	layers.push_back(std::make_shared<NeuralConvolutionalLayer>(
		CMAF_LOGISTIC, 1, 1, 1, 1, 1, 1, 1));
	layers.push_back(std::make_shared<NeuralConvolutionalLayer>(
		CMAF_LOGISTIC, 1, 1, 1, 1, 1, 1, 1));
	layers.push_back(std::make_shared<NeuralConvolutionalLayer>(
		CMAF_LOGISTIC, 1, 1, 1, 1, 1, 1, 1, RE_NORMAL));
	layers.push_back(std::make_shared<NeuralLinearLayer>(4));
	auto network = std::make_shared<NeuralNetwork>(layers);
	network->put("seed", seed);

	network->connect(0,2);
	network->connect(1,2);
	network->connect(2,3);
	network->connect(2,4);
	network->connect(3,5);
	network->connect(4,5);

	network->initialize_neural_network();
	network->set_l1_coefficient(0.03);
	network->set_l2_coefficient(0.01);
	EXPECT_NEAR(network->check_gradients(), 0.0, tolerance);

}

/** tests a neural network on the binary XOR problem */
TEST(NeuralNetwork, binary_classification)
{
	int32_t seed = 127;

	SGMatrix<float64_t> inputs_matrix(2,4);
	SGVector<float64_t> targets_vector(4);
	inputs_matrix(0,0) = -1.0;
	inputs_matrix(1,0) = -1.0;
	targets_vector[0] = -1.0;

	inputs_matrix(0,1) = -1.0;
	inputs_matrix(1,1) = 1.0;
	targets_vector[1] = 1.0;

	inputs_matrix(0,2) = 1.0;
	inputs_matrix(1,2) = -1.0;
	targets_vector[2] = 1.0;

	inputs_matrix(0,3) = 1.0;
	inputs_matrix(1,3) = 1.0;
	targets_vector[3] = -1.0;

	auto features =
		std::make_shared<DenseFeatures<float64_t>>(inputs_matrix);

	auto labels = std::make_shared<BinaryLabels>(targets_vector);

	std::vector<std::shared_ptr<NeuralLayer>> layers;
	layers.push_back(std::make_shared<NeuralInputLayer>(2));
	layers.push_back(std::make_shared<NeuralLogisticLayer>(2));
	layers.push_back(std::make_shared<NeuralLogisticLayer>(1));

	auto network = std::make_shared<NeuralNetwork>(layers);
	network->put("seed", seed);
	network->put("sigma", 0.1);

	network->set_epsilon(1e-8);

	network->train(features, labels);

	auto predictions = network->apply_binary(features);

	for (int32_t i=0; i<4; i++)
		EXPECT_EQ(predictions->get_label(i), labels->get_label(i));

	EXPECT_NEAR(predictions->get_value(0), 0, 1e-8);
	EXPECT_NEAR(predictions->get_value(1), 1, 1e-8);
	EXPECT_NEAR(predictions->get_value(2), 1, 1e-8);
	EXPECT_NEAR(predictions->get_value(3), 0, 1e-8);
}

/** tests a neural network on the multiclass XOR problem (the binary xor problem
 * in multiclass form)
 */
TEST(NeuralNetwork, multiclass_classification)
{
	int32_t seed = 100;

	SGMatrix<float64_t> inputs_matrix(2,4);
	SGVector<float64_t> targets_vector(4);
	inputs_matrix(0,0) = -1.0;
	inputs_matrix(1,0) = -1.0;
	targets_vector[0] = 0.0;

	inputs_matrix(0,1) = -1.0;
	inputs_matrix(1,1) = 1.0;
	targets_vector[1] = 1.0;

	inputs_matrix(0,2) = 1.0;
	inputs_matrix(1,2) = -1.0;
	targets_vector[2] = 1.0;

	inputs_matrix(0,3) = 1.0;
	inputs_matrix(1,3) = 1.0;
	targets_vector[3] = 0.0;

	auto features =
		std::make_shared<DenseFeatures<float64_t>>(inputs_matrix);

	auto labels = std::make_shared<MulticlassLabels>(targets_vector);

	std::vector<std::shared_ptr<NeuralLayer>> layers;
	layers.push_back(std::make_shared<NeuralInputLayer>(2));
	layers.push_back(std::make_shared<NeuralLogisticLayer>(2));
	layers.push_back(std::make_shared<NeuralLogisticLayer>(2));

	auto network = std::make_shared<NeuralNetwork>(layers);
	network->put("seed", seed);
	network->put("sigma", 0.1);

	network->set_epsilon(1e-8);

	network->train(features, labels);

	auto predictions = network->apply_multiclass(features);

	for (int32_t i=0; i<4; i++)
		EXPECT_EQ(predictions->get_label(i), labels->get_label(i));

	for (int32_t i=0; i<4; i++)
	{
		SGVector<float64_t> confidences = predictions->get_multiclass_confidences(i);
		EXPECT_NEAR(confidences[0], 1.0-targets_vector[i], 1e-8);
		EXPECT_NEAR(confidences[1], targets_vector[i], 1e-8);
	}
}

/** tests a neural network on a very simple regression problem */
TEST(NeuralNetwork, regression)
{
	int32_t seed = 100;

	int32_t N = 20;
	SGMatrix<float64_t> inputs_matrix(1,N);
	SGVector<float64_t> targets_vector(N);

	for (int32_t i=0; i<N; i++)
	{
		inputs_matrix(0,i) = i;
		targets_vector[i] = i*i;
	}

	auto features =
		std::make_shared<DenseFeatures<float64_t>>(inputs_matrix);

	auto labels = std::make_shared<RegressionLabels>(targets_vector);

	std::vector<std::shared_ptr<NeuralLayer>> layers;
	layers.push_back(std::make_shared<NeuralInputLayer>(1));
	layers.push_back(std::make_shared<NeuralLogisticLayer>(20));
	layers.push_back(std::make_shared<NeuralLinearLayer>(1));

	auto network = std::make_shared<NeuralNetwork>(layers);
	network->put("seed", seed);
	network->put("sigma", 1e-6);

	network->set_epsilon(1e-6);

	network->train(features, labels);

	auto predictions = network->apply_regression(features);

	for (int32_t i=0; i<N; i++)
		EXPECT_NEAR(predictions->get_label(i), labels->get_label(i), 0.5);
}

/** tests a neural network (trained using gradient descent) on the binary XOR
 * problem
 */
TEST(NeuralNetwork, gradient_descent)
{
	int32_t seed = 100;

	SGMatrix<float64_t> inputs_matrix(2,4);
	SGVector<float64_t> targets_vector(4);
	inputs_matrix(0,0) = -1.0;
	inputs_matrix(1,0) = -1.0;
	targets_vector[0] = -1.0;

	inputs_matrix(0,1) = -1.0;
	inputs_matrix(1,1) = 1.0;
	targets_vector[1] = 1.0;

	inputs_matrix(0,2) = 1.0;
	inputs_matrix(1,2) = -1.0;
	targets_vector[2] = 1.0;

	inputs_matrix(0,3) = 1.0;
	inputs_matrix(1,3) = 1.0;
	targets_vector[3] = -1.0;

	auto features =
		std::make_shared<DenseFeatures<float64_t>>(inputs_matrix);

	auto labels = std::make_shared<BinaryLabels>(targets_vector);

	std::vector<std::shared_ptr<NeuralLayer>> layers;
	layers.push_back(std::make_shared<NeuralInputLayer>(2));
	layers.push_back(std::make_shared<NeuralLogisticLayer>(2));
	layers.push_back(std::make_shared<NeuralLogisticLayer>(1));

	auto network = std::make_shared<NeuralNetwork>(layers);
	network->put("seed", seed);
	network->put("sigma", 0.1);

	network->set_optimization_method(NNOM_GRADIENT_DESCENT);
	network->set_gd_learning_rate(10.0);
	network->set_epsilon(0.0);
	network->set_max_num_epochs(1000);

	network->train(features, labels);

	auto predictions = network->apply_binary(features);

	for (int32_t i=0; i<4; i++)
		EXPECT_EQ(predictions->get_label(i), labels->get_label(i));
}
