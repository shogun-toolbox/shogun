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

#include <shogun/base/init.h>

#include <shogun/mathematics/Math.h>
#include <shogun/features/DataGenerator.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/evaluation/MulticlassAccuracy.h>

#include <shogun/lib/DynamicObjectArray.h>
#include <shogun/neuralnets/NeuralNetwork.h>
#include <shogun/neuralnets/NeuralInputLayer.h>
#include <shogun/neuralnets/NeuralSoftmaxLayer.h>
#include <shogun/neuralnets/NeuralConvolutionalLayer.h>

using namespace shogun;

int main(int, char*[])
{
	init_shogun_with_defaults();
	
#ifdef HAVE_LAPACK // for CDataGenerator::generate_gaussian()
	
	// initialize the random number generator with a fixed seed, for repeatability
	CMath::init_random(10);
	
	// Prepare the training data
	const int width = 4;
	const int height = 4;
	const int num_channels = 2;
	const int num_features = width*height*num_channels;
	const int num_classes = 3;
	const int num_examples_per_class = 15;
	
	SGMatrix<float64_t> X;
	SGVector<float64_t> Y;
	try
	{
		X = CDataGenerator::generate_gaussians(
			num_examples_per_class,num_classes,num_features);
		
		Y = SGVector<float64_t>(num_classes*num_examples_per_class);
	}
	catch (ShogunException e)
	{
		// out of memory
		SG_SPRINT(e.get_exception_string());
		return 0;
	}
	
	for (int32_t i = 0; i < num_classes; i++)
		for (int32_t j = 0; j < num_examples_per_class; j++)
			Y[i*num_examples_per_class + j] = i;
	
	CDenseFeatures<float64_t>* features = new CDenseFeatures<float64_t>(X);
	CMulticlassLabels* labels = new CMulticlassLabels(Y);
	
	// prepare the layers
	CDynamicObjectArray* layers = new CDynamicObjectArray();
	
	// input layer
	layers->append_element(new CNeuralInputLayer(width,height,num_channels));
	
	// first convolutional layer: 3 feature maps, 3x3 masks, 2x2 max-pooling
	layers->append_element(new CNeuralConvolutionalLayer(
		CMAF_RECTIFIED_LINEAR, 3, 1,1, 2,2));
	
	// second convolutional layer: 5 feature maps, 3x3 masks
	layers->append_element(new CNeuralConvolutionalLayer(
		CMAF_RECTIFIED_LINEAR, 5, 1,1));
	
	// output layer
	layers->append_element(new CNeuralSoftmaxLayer(num_classes));
	
	// create and initialize the network
	CNeuralNetwork* network = new CNeuralNetwork(layers);
	network->quick_connect();
	network->initialize(0.1);
	
	// uncomment this line to enable info logging
	// network->io->set_loglevel(MSG_INFO);
	
	// train using default parameters
	network->set_labels(labels);
	network->train(features);
	
	// evaluate
	CMulticlassLabels* predictions = network->apply_multiclass(features);
	CMulticlassAccuracy* evaluator = new CMulticlassAccuracy();
	float64_t accuracy = evaluator->evaluate(predictions, labels);
	
	SG_SINFO("Accuracy = %f %\n", accuracy*100);
	
	// Clean up
	SG_UNREF(network);
	SG_UNREF(features);
	SG_UNREF(predictions);
	SG_UNREF(evaluator);
	
#endif
	
	exit_shogun();
	return 0;
}
