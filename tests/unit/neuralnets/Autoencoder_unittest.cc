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
#include <shogun/neuralnets/Autoencoder.h>
#include <shogun/neuralnets/NeuralInputLayer.h>
#include <shogun/neuralnets/NeuralRectifiedLinearLayer.h>
#include <shogun/neuralnets/NeuralLogisticLayer.h>
#include <shogun/neuralnets/NeuralConvolutionalLayer.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/UniformRealDistribution.h>

#include <random>

using namespace shogun;

TEST(Autoencoder, train)
{
	int32_t seed = 100;
	int32_t num_features = 10;
	int32_t num_examples = 100;
	int32_t num_hid = 10;

	std::mt19937_64 prng(seed);
	UniformRealDistribution<float64_t> uniform_real_dist(-1.0, 1.0);
	SGMatrix<float64_t> data(num_features, num_examples);
	for (int32_t i=0; i<num_features*num_examples; i++)
		data[i] = uniform_real_dist(prng);

	// the constructor of Autoencoder initializes the weights of the hidden
	// layer (which is random) before seed can be put
	auto hidden_layer = std::make_shared<NeuralRectifiedLinearLayer>(num_hid);
	auto decoding_layer = std::make_shared<NeuralLinearLayer>(num_features);
	hidden_layer->put("seed", seed);
	decoding_layer->put("seed", seed);
	Autoencoder ae(num_features, hidden_layer, decoding_layer);
	ae.put("seed", seed);

	auto features = std::make_shared<DenseFeatures<float64_t>>(data);

	ae.train(features);

	auto reconstructed = ae.reconstruct(features);
	SGMatrix<float64_t> reconstructed_data = reconstructed->get_feature_matrix();

	float64_t avg_diff = 0;
	for (int32_t i=0; i<num_features*num_examples; i++)
		avg_diff += Math::abs(reconstructed_data[i]-data[i])/(num_examples*num_features);

	EXPECT_NEAR(0.0, avg_diff, 1e-6);



}

/** Tests gradients computed using backpropagation against gradients computed
 * by numerical approximation. Uses a NeuralLinearLayer-based contractive
 * autoencoder.
 */
TEST(Autoencoder, contractive_linear)
{
	int32_t seed = 10;
	float64_t tolerance = 1e-9;

	auto hidden_layer = std::make_shared<NeuralLinearLayer>(15);
	auto decoding_layer = std::make_shared<NeuralLinearLayer>(10);
	hidden_layer->put("seed", seed);
	decoding_layer->put("seed", seed);
	Autoencoder ae(10, hidden_layer, decoding_layer);
	ae.put("seed", seed);
	ae.set_contraction_coefficient(10.0);

	EXPECT_NEAR(ae.check_gradients(), 0.0, tolerance);
}

/** Tests gradients computed using backpropagation against gradients computed
 * by numerical approximation. Uses a NeuralRectifiedLinearLayer-based contractive
 * autoencoder.
 */
TEST(Autoencoder, contractive_rectified_linear)
{
	int32_t seed = 10;
	float64_t tolerance = 1e-9;

	auto hidden_layer = std::make_shared<NeuralRectifiedLinearLayer>(15);
	auto decoding_layer = std::make_shared<NeuralLinearLayer>(10);
	hidden_layer->put("seed", seed);
	decoding_layer->put("seed", seed);
	Autoencoder ae(10, hidden_layer, decoding_layer);
	ae.put("seed", seed);
	ae.set_contraction_coefficient(10.0);

	EXPECT_NEAR(ae.check_gradients(), 0.0, tolerance);
}

/** Tests gradients computed using backpropagation against gradients computed
 * by numerical approximation. Uses a NeuralLogisticLayer-based contractive
 * autoencoder.
 */
TEST(Autoencoder, contractive_logistic)
{
	int32_t seed = 10;
	float64_t tolerance = 1e-6;

	auto hidden_layer = std::make_shared<NeuralLogisticLayer>(15);
	auto decoding_layer = std::make_shared<NeuralLinearLayer>(10);
	hidden_layer->put("seed", seed);
	decoding_layer->put("seed", seed);
	Autoencoder ae(10, hidden_layer, decoding_layer);
	ae.put("seed", seed);
	ae.initialize_neural_network();

	ae.set_contraction_coefficient(1.0);

	EXPECT_NEAR(ae.check_gradients(), 0.0, tolerance);
}

/** Tests gradients computed using backpropagation against gradients computed
 * by numerical approximation. Uses a NeuralConvolutionalLayer-based autoencoder.
 */
TEST(Autoencoder, convolutional)
{
	const int32_t seed = 10;
	const int32_t w = 12;
	const int32_t h = 10;

	float64_t tolerance = 1e-9;

	auto hidden_layer = std::make_shared<NeuralConvolutionalLayer>(CMAF_IDENTITY, 2, 1,1, 1,1, 1,1);
	auto decoding_layer = std::make_shared<NeuralConvolutionalLayer>(CMAF_IDENTITY, 3, 1,1, 1,1, 1,1);
	hidden_layer->put("seed", seed);
	decoding_layer->put("seed", seed);
	Autoencoder ae(w,h,3,hidden_layer,decoding_layer);
	ae.put("seed", seed);

	EXPECT_NEAR(ae.check_gradients(), 0.0, tolerance);
}

/** Tests gradients computed using backpropagation against gradients computed
 * by numerical approximation. Uses a NeuralConvolutionalLayer-based autoencoder.
 */
TEST(Autoencoder, convolutional_with_pooling)
{
	const int32_t seed = 10;
	const int32_t w = 12;
	const int32_t h = 10;

	float64_t tolerance = 1e-9;

	auto hidden_layer = std::make_shared<NeuralConvolutionalLayer>(CMAF_IDENTITY, 2, 1,1, 3,2, 1,1);
	auto decoding_layer = std::make_shared<NeuralConvolutionalLayer>(CMAF_IDENTITY, 3, 1,1, 1,1, 1,1);
	hidden_layer->put("seed", seed);
	decoding_layer->put("seed", seed);
	Autoencoder ae(w,h,3,hidden_layer,decoding_layer);
	ae.put("seed", seed);

	EXPECT_NEAR(ae.check_gradients(), 0.0, tolerance);
}

/** Tests gradients computed using backpropagation against gradients computed
 * by numerical approximation. Uses a NeuralConvolutionalLayer-based autoencoder.
 */
TEST(Autoencoder, convolutional_with_stride)
{
	const int32_t seed = 10;
	const int32_t w = 12;
	const int32_t h = 10;

	float64_t tolerance = 1e-9;
	auto hidden_layer = std::make_shared<NeuralConvolutionalLayer>(CMAF_IDENTITY, 2, 1,1, 1,1, 3,2);
	auto decoding_layer = std::make_shared<NeuralConvolutionalLayer>(CMAF_IDENTITY, 3, 1,1, 1,1, 1,1);
	hidden_layer->put("seed", seed);
	decoding_layer->put("seed", seed);
	Autoencoder ae(w,h,3,hidden_layer,decoding_layer);
	ae.put("seed", seed);

	EXPECT_NEAR(ae.check_gradients(), 0.0, tolerance);
}

/** Tests gradients computed using backpropagation against gradients computed
 * by numerical approximation. Uses a NeuralConvolutionalLayer-based autoencoder.
 */
TEST(Autoencoder, convolutional_with_stride_and_pooling)
{
	const int32_t seed = 10;
	const int32_t w = 16;
	const int32_t h = 16;

	float64_t tolerance = 1e-9;

	auto hidden_layer = std::make_shared<NeuralConvolutionalLayer>(CMAF_IDENTITY, 2, 1,1, 2,2, 2,2);
	auto decoding_layer = std::make_shared<NeuralConvolutionalLayer>(CMAF_IDENTITY, 3, 1,1, 1,1, 1,1);
	hidden_layer->put("seed", seed);
	decoding_layer->put("seed", seed);
	Autoencoder ae(w,h,3,hidden_layer,decoding_layer);
	ae.put("seed", seed);

	EXPECT_NEAR(ae.check_gradients(), 0.0, tolerance);
}
