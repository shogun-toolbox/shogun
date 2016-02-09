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
#include <shogun/neuralnets/Autoencoder.h>
#include <shogun/neuralnets/NeuralInputLayer.h>
#include <shogun/neuralnets/NeuralRectifiedLinearLayer.h>
#include <shogun/neuralnets/NeuralLogisticLayer.h>
#include <shogun/neuralnets/NeuralConvolutionalLayer.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/mathematics/Math.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(Autoencoder, train)
{
	CMath::init_random(100);

	int32_t num_features = 10;
	int32_t num_examples = 100;
	int32_t num_hid = 10;

	SGMatrix<float64_t> data(num_features, num_examples);
	for (int32_t i=0; i<num_features*num_examples; i++)
		data[i] = CMath::random(-1.0,1.0);

	CAutoencoder ae(num_features, new CNeuralRectifiedLinearLayer(num_hid));

	CDenseFeatures<float64_t>* features = new CDenseFeatures<float64_t>(data);

	ae.train(features);

	CDenseFeatures<float64_t>* reconstructed = ae.reconstruct(features);
	SGMatrix<float64_t> reconstructed_data = reconstructed->get_feature_matrix();

	float64_t avg_diff = 0;
	for (int32_t i=0; i<num_features*num_examples; i++)
		avg_diff += CMath::abs(reconstructed_data[i]-data[i])/(num_examples*num_features);

	EXPECT_NEAR(0.0, avg_diff, 1e-6);

	SG_UNREF(features);
	SG_UNREF(reconstructed);
}

/** Tests gradients computed using backpropagation against gradients computed
 * by numerical approximation. Uses a CNeuralLinearLayer-based contractive
 * autoencoder.
 */
TEST(Autoencoder, contractive_linear)
{
	float64_t tolerance = 1e-9;

	CMath::init_random(10);

	CAutoencoder ae(10, new CNeuralLinearLayer(15));

	ae.set_contraction_coefficient(10.0);

	EXPECT_NEAR(ae.check_gradients(), 0.0, tolerance);
}

/** Tests gradients computed using backpropagation against gradients computed
 * by numerical approximation. Uses a CNeuralRectifiedLinearLayer-based contractive
 * autoencoder.
 */
TEST(Autoencoder, contractive_rectified_linear)
{
	float64_t tolerance = 1e-9;

	CMath::init_random(10);

	CAutoencoder ae(10, new CNeuralRectifiedLinearLayer(15));

	ae.set_contraction_coefficient(10.0);

	EXPECT_NEAR(ae.check_gradients(), 0.0, tolerance);
}

/** Tests gradients computed using backpropagation against gradients computed
 * by numerical approximation. Uses a CNeuralLogisticLayer-based contractive
 * autoencoder.
 */
TEST(Autoencoder, contractive_logistic)
{
	float64_t tolerance = 1e-6;

	CMath::init_random(10);

	CAutoencoder ae(10, new CNeuralLogisticLayer(15));
	ae.initialize_neural_network();

	ae.set_contraction_coefficient(1.0);

	EXPECT_NEAR(ae.check_gradients(), 0.0, tolerance);
}

/** Tests gradients computed using backpropagation against gradients computed
 * by numerical approximation. Uses a CNeuralConvolutionalLayer-based autoencoder.
 */
TEST(Autoencoder, convolutional)
{
	const int32_t w = 12;
	const int32_t h = 10;

	float64_t tolerance = 1e-9;

	CMath::init_random(10);

	CAutoencoder ae(w,h,3,
		new CNeuralConvolutionalLayer(CMAF_IDENTITY, 2, 1,1, 1,1, 1,1),
		new CNeuralConvolutionalLayer(CMAF_IDENTITY, 3, 1,1, 1,1, 1,1));

	EXPECT_NEAR(ae.check_gradients(), 0.0, tolerance);
}

/** Tests gradients computed using backpropagation against gradients computed
 * by numerical approximation. Uses a CNeuralConvolutionalLayer-based autoencoder.
 */
TEST(Autoencoder, convolutional_with_pooling)
{
	const int32_t w = 12;
	const int32_t h = 10;

	float64_t tolerance = 1e-9;

	CMath::init_random(10);

	CAutoencoder ae(w,h,3,
		new CNeuralConvolutionalLayer(CMAF_IDENTITY, 2, 1,1, 3,2, 1,1),
		new CNeuralConvolutionalLayer(CMAF_IDENTITY, 3, 1,1, 1,1, 1,1));

	EXPECT_NEAR(ae.check_gradients(), 0.0, tolerance);
}

/** Tests gradients computed using backpropagation against gradients computed
 * by numerical approximation. Uses a CNeuralConvolutionalLayer-based autoencoder.
 */
TEST(Autoencoder, convolutional_with_stride)
{
	const int32_t w = 12;
	const int32_t h = 10;

	float64_t tolerance = 1e-9;

	CMath::init_random(10);

	CAutoencoder ae(w,h,3,
		new CNeuralConvolutionalLayer(CMAF_IDENTITY, 2, 1,1, 1,1, 3,2),
		new CNeuralConvolutionalLayer(CMAF_IDENTITY, 3, 1,1, 1,1, 1,1));

	EXPECT_NEAR(ae.check_gradients(), 0.0, tolerance);
}

/** Tests gradients computed using backpropagation against gradients computed
 * by numerical approximation. Uses a CNeuralConvolutionalLayer-based autoencoder.
 */
TEST(Autoencoder, convolutional_with_stride_and_pooling)
{
	const int32_t w = 16;
	const int32_t h = 16;

	float64_t tolerance = 1e-9;

	CMath::init_random(10);

	CAutoencoder ae(w,h,3,
		new CNeuralConvolutionalLayer(CMAF_IDENTITY, 2, 1,1, 2,2, 2,2),
		new CNeuralConvolutionalLayer(CMAF_IDENTITY, 3, 1,1, 1,1, 1,1));

	EXPECT_NEAR(ae.check_gradients(), 0.0, tolerance);
}
