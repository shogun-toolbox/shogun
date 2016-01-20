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

#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/lib/DynamicObjectArray.h>
#include <shogun/neuralnets/NeuralNetwork.h>
#include <shogun/neuralnets/NeuralInputLayer.h>
#include <shogun/neuralnets/NeuralLogisticLayer.h>
#include <shogun/neuralnets/NeuralSoftmaxLayer.h>
#include <shogun/neuralnets/NeuralRectifiedLinearLayer.h>
#include <shogun/neuralnets/NeuralConvolutionalLayer.h>
#include <shogun/neuralnets/NeuralLayers.h>
#include <gtest/gtest.h>

using namespace shogun;

/** Tests gradients computed using backpropagation against gradients computed
 * by numerical approximation. Uses a CNeuralLinearLayer-based network.
 */
TEST(NeuralNetwork, backpropagation_linear)
{
	float64_t tolerance = 1e-9;

	CMath::init_random(10);

	CDynamicObjectArray* layers = new CDynamicObjectArray();
	layers->append_element(new CNeuralInputLayer(5));
	layers->append_element(new CNeuralInputLayer(7));
	layers->append_element(new CNeuralLinearLayer(3));
	layers->append_element(new CNeuralLinearLayer(6));
	layers->append_element(new CNeuralLinearLayer(5));
	layers->append_element(new CNeuralLinearLayer(4));
	CNeuralNetwork* network = new CNeuralNetwork(layers);

	network->connect(0,2);
	network->connect(1,2);
	network->connect(2,3);
	network->connect(2,4);
	network->connect(3,5);
	network->connect(4,5);

	network->initialize_neural_network();
	network->l2_coefficient = 0.01;
	network->l1_coefficient = 0.03;

	EXPECT_NEAR(network->check_gradients(), 0.0, tolerance);
	SG_UNREF(network);
}

/** Tests neural layers builder
 */
TEST(NeuralNetwork, neural_layers_builder)
{
	float64_t tolerance = 1e-9;

	CMath::init_random(10);

	CNeuralLayers* layers = new CNeuralLayers();
	layers->input(5)
	      ->input(7)
	      ->linear(3)
	      ->linear(6)
	      ->linear(5)
	      ->linear(4);
	CNeuralNetwork* network = new CNeuralNetwork(layers->done());

	network->connect(0,2);
	network->connect(1,2);
	network->connect(2,3);
	network->connect(2,4);
	network->connect(3,5);
	network->connect(4,5);

	network->initialize_neural_network();
	network->l2_coefficient = 0.01;
	network->l1_coefficient = 0.03;

	EXPECT_NEAR(network->check_gradients(), 0.0, tolerance);
	SG_UNREF(network);
	SG_UNREF(layers);
}


/** Tests gradients computed using backpropagation against gradients computed
 * by numerical approximation. Uses a CNeuralLogisticLayer-based network.
 */
TEST(NeuralNetwork, backpropagation_logistic)
{
	float64_t tolerance = 1e-9;

	CMath::init_random(10);

	CDynamicObjectArray* layers = new CDynamicObjectArray();
	layers->append_element(new CNeuralInputLayer(5));
	layers->append_element(new CNeuralInputLayer(7));
	layers->append_element(new CNeuralLogisticLayer(3));
	layers->append_element(new CNeuralLogisticLayer(6));
	layers->append_element(new CNeuralLogisticLayer(5));
	layers->append_element(new CNeuralLogisticLayer(4));
	CNeuralNetwork* network = new CNeuralNetwork(layers);

	network->connect(0,2);
	network->connect(1,2);
	network->connect(2,3);
	network->connect(2,4);
	network->connect(3,5);
	network->connect(4,5);

	network->initialize_neural_network();
	network->l1_coefficient = 0.03;
	network->l2_coefficient = 0.01;
	EXPECT_NEAR(network->check_gradients(), 0.0, tolerance);
	SG_UNREF(network);
}

/** Tests gradients computed using backpropagation against gradients computed
 * by numerical approximation. Uses a CNeuralSoftmaxLayer-based network.
 */
TEST(NeuralNetwork, backpropagation_softmax)
{
	float64_t tolerance = 1e-9;

	CMath::init_random(10);

	CDynamicObjectArray* layers = new CDynamicObjectArray();
	layers->append_element(new CNeuralInputLayer(5));
	layers->append_element(new CNeuralInputLayer(7));
	layers->append_element(new CNeuralLinearLayer(3));
	layers->append_element(new CNeuralLinearLayer(6));
	layers->append_element(new CNeuralLinearLayer(5));
	layers->append_element(new CNeuralSoftmaxLayer(4));
	CNeuralNetwork* network = new CNeuralNetwork(layers);

	network->connect(0,2);
	network->connect(1,2);
	network->connect(2,3);
	network->connect(2,4);
	network->connect(3,5);
	network->connect(4,5);

	network->initialize_neural_network();
	network->l1_coefficient = 0.03;
	network->l2_coefficient = 0.01;
	EXPECT_NEAR(network->check_gradients(), 0.0, tolerance);
	SG_UNREF(network);
}

/** Tests gradients computed using backpropagation against gradients computed
 * by numerical approximation. Uses a CNeuralRectifiedLinearLayer-based network.
 */
TEST(NeuralNetwork, backpropagation_rectified_linear)
{
	float64_t tolerance = 1e-9;

	CMath::init_random(10);

	CDynamicObjectArray* layers = new CDynamicObjectArray();
	layers->append_element(new CNeuralInputLayer(5));
	layers->append_element(new CNeuralInputLayer(7));
	layers->append_element(new CNeuralRectifiedLinearLayer(3));
	layers->append_element(new CNeuralRectifiedLinearLayer(6));
	layers->append_element(new CNeuralRectifiedLinearLayer(5));
	layers->append_element(new CNeuralLinearLayer(4));
	CNeuralNetwork* network = new CNeuralNetwork(layers);

	network->connect(0,2);
	network->connect(1,2);
	network->connect(2,3);
	network->connect(2,4);
	network->connect(3,5);
	network->connect(4,5);

	network->initialize_neural_network();
	network->l1_coefficient = 0.03;
	network->l2_coefficient = 0.01;
	EXPECT_NEAR(network->check_gradients(), 0.0, tolerance);
	SG_UNREF(network);
}

/** Tests gradients computed using backpropagation against gradients computed
 * by numerical approximation. Uses a CNeuralConvolutionalLayer-based network.
 */
TEST(NeuralNetwork, backpropagation_convolutional)
{
	float64_t tolerance = 1e-9;

	CMath::init_random(10);

	CDynamicObjectArray* layers = new CDynamicObjectArray();
	layers->append_element(new CNeuralInputLayer(6,4));
	layers->append_element(new CNeuralInputLayer(6,4));
	layers->append_element(new CNeuralConvolutionalLayer(
		CMAF_LOGISTIC, 1, 1, 1, 1, 1, 1, 1));
	layers->append_element(new CNeuralConvolutionalLayer(
		CMAF_LOGISTIC, 1, 1, 1, 1, 1, 1, 1));
	layers->append_element(new CNeuralConvolutionalLayer(
		CMAF_LOGISTIC, 1, 1, 1, 1, 1, 1, 1, RE_NORMAL));
	layers->append_element(new CNeuralLinearLayer(4));
	CNeuralNetwork* network = new CNeuralNetwork(layers);

	network->connect(0,2);
	network->connect(1,2);
	network->connect(2,3);
	network->connect(2,4);
	network->connect(3,5);
	network->connect(4,5);

	network->initialize_neural_network();
	network->l1_coefficient = 0.03;
	network->l2_coefficient = 0.01;
	EXPECT_NEAR(network->check_gradients(), 0.0, tolerance);
	SG_UNREF(network);
}

/** tests a neural network on the binary XOR problem */
TEST(NeuralNetwork, binary_classification)
{
	CMath::init_random(100);

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

	CDenseFeatures<float64_t>* features =
		new CDenseFeatures<float64_t>(inputs_matrix);

	CBinaryLabels* labels = new CBinaryLabels(targets_vector);

	CDynamicObjectArray* layers = new CDynamicObjectArray();
	layers->append_element(new CNeuralInputLayer(2));
	layers->append_element(new CNeuralLogisticLayer(2));
	layers->append_element(new CNeuralLogisticLayer(1));

	CNeuralNetwork* network = new CNeuralNetwork(layers);
	network->quick_connect();
	network->initialize_neural_network(0.1);

	network->epsilon = 1e-8;

	network->set_labels(labels);
	network->train(features);

	CBinaryLabels* predictions = network->apply_binary(features);

	for (int32_t i=0; i<4; i++)
		EXPECT_EQ(predictions->get_label(i), labels->get_label(i));

	EXPECT_NEAR(predictions->get_value(0), 0, 1e-8);
	EXPECT_NEAR(predictions->get_value(1), 1, 1e-8);
	EXPECT_NEAR(predictions->get_value(2), 1, 1e-8);
	EXPECT_NEAR(predictions->get_value(3), 0, 1e-8);

	SG_UNREF(network);
	SG_UNREF(features);
	SG_UNREF(predictions);
}

/** tests a neural network on the multiclass XOR problem (the binary xor problem
 * in multiclass form)
 */
TEST(NeuralNetwork, multiclass_classification)
{
	CMath::init_random(100);

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

	CDenseFeatures<float64_t>* features =
		new CDenseFeatures<float64_t>(inputs_matrix);

	CMulticlassLabels* labels = new CMulticlassLabels(targets_vector);

	CDynamicObjectArray* layers = new CDynamicObjectArray();
	layers->append_element(new CNeuralInputLayer(2));
	layers->append_element(new CNeuralLogisticLayer(2));
	layers->append_element(new CNeuralLogisticLayer(2));

	CNeuralNetwork* network = new CNeuralNetwork(layers);
	network->quick_connect();
	network->initialize_neural_network(0.1);

	network->epsilon = 1e-8;

	network->set_labels(labels);
	network->train(features);

	CMulticlassLabels* predictions = network->apply_multiclass(features);

	for (int32_t i=0; i<4; i++)
		EXPECT_EQ(predictions->get_label(i), labels->get_label(i));

	for (int32_t i=0; i<4; i++)
	{
		SGVector<float64_t> confidences = predictions->get_multiclass_confidences(i);
		EXPECT_NEAR(confidences[0], 1.0-targets_vector[i], 1e-8);
		EXPECT_NEAR(confidences[1], targets_vector[i], 1e-8);
	}

	SG_UNREF(network);
	SG_UNREF(features);
	SG_UNREF(predictions);
}

/** tests a neural network on a very simple regression problem */
TEST(NeuralNetwork, regression)
{
	CMath::init_random(100);

	int32_t N = 20;
	SGMatrix<float64_t> inputs_matrix(1,N);
	SGVector<float64_t> targets_vector(N);

	for (int32_t i=0; i<N; i++)
	{
		inputs_matrix(0,i) = i;
		targets_vector[i] = i*i;
	}

	CDenseFeatures<float64_t>* features =
		new CDenseFeatures<float64_t>(inputs_matrix);

	CRegressionLabels* labels = new CRegressionLabels(targets_vector);

	CDynamicObjectArray* layers = new CDynamicObjectArray();
	layers->append_element(new CNeuralInputLayer(1));
	layers->append_element(new CNeuralLogisticLayer(20));
	layers->append_element(new CNeuralLinearLayer(1));

	CNeuralNetwork* network = new CNeuralNetwork(layers);
	network->quick_connect();
	network->initialize_neural_network(1e-6);

	network->epsilon = 1e-6;

	network->set_labels(labels);
	network->train(features);

	CRegressionLabels* predictions = network->apply_regression(features);

	for (int32_t i=0; i<N; i++)
		EXPECT_NEAR(predictions->get_label(i), labels->get_label(i), 0.5);

	SG_UNREF(network);
	SG_UNREF(features);
	SG_UNREF(predictions);
}

/** tests a neural network (trained using gradient descent) on the binary XOR
 * problem
 */
TEST(NeuralNetwork, gradient_descent)
{
	CMath::init_random(100);

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

	CDenseFeatures<float64_t>* features =
		new CDenseFeatures<float64_t>(inputs_matrix);

	CBinaryLabels* labels = new CBinaryLabels(targets_vector);

	CDynamicObjectArray* layers = new CDynamicObjectArray();
	layers->append_element(new CNeuralInputLayer(2));
	layers->append_element(new CNeuralLogisticLayer(2));
	layers->append_element(new CNeuralLogisticLayer(1));

	CNeuralNetwork* network = new CNeuralNetwork(layers);
	network->quick_connect();
	network->initialize_neural_network(0.1);

	network->optimization_method = NNOM_GRADIENT_DESCENT;
	network->gd_learning_rate = 10.0;
	network->epsilon = 0.0;
	network->max_num_epochs = 1000;
	network->rms_epsilon = 1.0;
	network->rms_decay_factor = 0.99;

	network->set_labels(labels);
	network->train(features);

	CBinaryLabels* predictions = network->apply_binary(features);

	for (int32_t i=0; i<4; i++)
		EXPECT_EQ(predictions->get_label(i), labels->get_label(i));

	SG_UNREF(network);
	SG_UNREF(features);
	SG_UNREF(predictions);
}

TEST(NeuralNetwork, rms_prop)
{
	CMath::init_random(100);

	int32_t N = 20;
	SGMatrix<float64_t> inputs_matrix(1,N);
	SGVector<float64_t> targets_vector(N);

	for (int32_t i=0; i<N; i++)
	{
		inputs_matrix(0,i) = i;
		targets_vector[i] = i*i;
  	}

	CDenseFeatures<float64_t>* features =
		new CDenseFeatures<float64_t>(inputs_matrix);

	CBinaryLabels* labels = new CBinaryLabels(targets_vector);

	CDynamicObjectArray* layers = new CDynamicObjectArray();
	layers->append_element(new CNeuralInputLayer(1));
	layers->append_element(new CNeuralLogisticLayer(5));
	layers->append_element(new CNeuralLogisticLayer(1));

	CNeuralNetwork* network = new CNeuralNetwork(layers);
	network->quick_connect();
	network->initialize_neural_network(0.1);

	network->optimization_method = NNOM_GRADIENT_DESCENT;
	network->gd_learning_rate = 10.0;
	network->epsilon = 0.0;
	network->max_num_epochs = 1000;
	network->rms_epsilon = 1e-6;
	network->rms_decay_factor = 0.9;

	network->set_labels(labels);
	network->train(features);

	CBinaryLabels* predictions = network->apply_binary(features);

	for (int32_t i=0; i<N; i++)
		EXPECT_EQ(predictions->get_label(i), labels->get_label(i));

	SG_UNREF(network);
	SG_UNREF(features);
  	SG_UNREF(predictions);
}
