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
#include <shogun/neuralnets/DeepAutoencoder.h>
#include <shogun/neuralnets/NeuralInputLayer.h>
#include <shogun/neuralnets/NeuralRectifiedLinearLayer.h>
#include <shogun/neuralnets/NeuralLogisticLayer.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/DynamicObjectArray.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/mathematics/Math.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(DeepAutoencoder, pre_train)
{
	CMath::init_random(10);
	
	int32_t num_features = 10;
	int32_t num_examples = 100;
	
	SGMatrix<float64_t> data(num_features, num_examples);
	for (int32_t i=0; i<num_features*num_examples; i++)
		data[i] = CMath::random(-1.0,1.0);
	
	CDynamicObjectArray* layers = new CDynamicObjectArray();
	layers->append_element(new CNeuralInputLayer(num_features));
	layers->append_element(new CNeuralLogisticLayer(12));
	layers->append_element(new CNeuralLogisticLayer(15));
	layers->append_element(new CNeuralLogisticLayer(12));
	layers->append_element(new CNeuralLinearLayer(num_features));
	
	CDeepAutoencoder ae(layers);
	
	CDenseFeatures<float64_t>* features = new CDenseFeatures<float64_t>(data);
	
	ae.pre_train(features);
	
	CDenseFeatures<float64_t>* reconstructed = ae.reconstruct(features);
	SGMatrix<float64_t> reconstructed_data = reconstructed->get_feature_matrix();
	
	float64_t avg_diff = 0;
	for (int32_t i=0; i<num_features*num_examples; i++)
		avg_diff += CMath::abs(reconstructed_data[i]-data[i])/(num_examples*num_features);
	
	EXPECT_NEAR(0.0, avg_diff, 0.01);
	
	SG_UNREF(features);
	SG_UNREF(reconstructed);
}

TEST(DeepAutoencoder, convert_to_neural_network)
{
	CMath::init_random(100);

	CDynamicObjectArray* layers = new CDynamicObjectArray();
	layers->append_element(new CNeuralInputLayer(10));
	layers->append_element(new CNeuralLogisticLayer(12));
	layers->append_element(new CNeuralLogisticLayer(15));
	layers->append_element(new CNeuralLogisticLayer(12));
	layers->append_element(new CNeuralLinearLayer(10));
	
	CDeepAutoencoder ae(layers);

	CNeuralNetwork* nn = ae.convert_to_neural_network();

	SGMatrix<float64_t> x(10, 3);
	for (int32_t i=0; i<x.num_rows*x.num_cols; i++)
		x[i] = CMath::random(0.0,1.0);

	CDenseFeatures<float64_t> f(x);
	
	CDenseFeatures<float64_t>* f_transformed_nn = nn->transform(&f);
	CDenseFeatures<float64_t>* f_transformed_ae = ae.transform(&f);
	

	SGMatrix<float64_t> x_transformed_ae = 
		f_transformed_ae->get_feature_matrix();

	SGMatrix<float64_t> x_transformed_nn = 
		f_transformed_nn->get_feature_matrix();
	
		
	for (int32_t i=0; i< x_transformed_ae.num_rows*x_transformed_ae.num_cols; i++)
		EXPECT_NEAR(x_transformed_ae[i], x_transformed_nn[i], 1e-15);

	SG_UNREF(nn);
	SG_UNREF(f_transformed_ae);
	SG_UNREF(f_transformed_nn);
}
