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

#include <shogun/neuralnets/DeepAutoencoder.h>
#include <shogun/neuralnets/NeuralLayers.h>

using namespace shogun;

int main(int, char*[])
{
	init_shogun_with_defaults();
	
#ifdef HAVE_LAPACK // for CDataGenerator::generate_gaussian()
	
	// initialize the random number generator with a fixed seed, for repeatability
	CMath::init_random(10);
	
	// Prepare the training data
	const int num_features = 20;
	const int num_classes = 4;
	const int num_examples_per_class = 20;
	
	SGMatrix<float64_t> X = CDataGenerator::generate_gaussians(
		num_examples_per_class,num_classes,num_features);
	
	CDenseFeatures<float64_t>* features = new CDenseFeatures<float64_t>(X);
	
	// Create a deep autoencoder
	CNeuralLayers* layers = new CNeuralLayers();
	layers
		->input(num_features)
		->rectified_linear(10)->rectified_linear(5)->rectified_linear(10)
		->linear(num_features);
	CDeepAutoencoder* ae = new CDeepAutoencoder(layers->done());
	
	// uncomment this line to enable info logging
	ae->io->set_loglevel(MSG_INFO);
	
	// pre-train
	ae->pt_epsilon.set_const(1e-12);
	ae->pre_train(features);
	
	// fine-tune
	ae->train(features);
	
	// reconstruct the data
	CDenseFeatures<float64_t>* reconstructions = ae->reconstruct(features);
	SGMatrix<float64_t> X_reconstructed = reconstructions->get_feature_matrix();
	
	// find the average difference between the data and the reconstructions
	float64_t avg_diff = 0;
	int32_t N = X.num_rows*X.num_cols;
	for (int32_t i=0; i<N; i++)
		avg_diff += CMath::abs(X[i]-X_reconstructed[i])/CMath::abs(X[i]);
	avg_diff /= N;
	
	SG_SINFO("Average difference = %f %\n", avg_diff*100);
	
	// Clean up
	SG_UNREF(ae);
	SG_UNREF(layers);
	SG_UNREF(features);
	SG_UNREF(reconstructions);
	
#endif
	
	exit_shogun();
	return 0;
}
