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
#ifdef HAVE_EIGEN3
#include <shogun/mathematics/Math.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/mathematics/Statistics.h>

#include <shogun/neuralnets/DeepBeliefNetwork.h>

using namespace shogun;

int main(int, char*[])
{
	init_shogun_with_defaults();

	// initialize the random number generator with a fixed seed, for repeatability
	CMath::init_random(10);

	// Prepare the training data
	const int num_features = 5;
	const int num_examples= 50;

	SGVector<float64_t> means;
	SGMatrix<float64_t> X;
	try
	{
		means = SGVector<float64_t>(num_features);
		X = SGMatrix<float64_t>(num_features, num_examples);

	}
	catch (ShogunException e)
	{
		// out of memory
		SG_SPRINT(e.get_exception_string());
		return 0;
	}

	for (int32_t i=0; i<num_features; i++)
			means[i] = CMath::random(-1.0,1.0);

	for (int32_t i=0; i<num_features; i++)
			for (int32_t j=0; j<num_examples; j++)
				X(i,j) = CMath::normal_random(means[i], 1.0);

	CDenseFeatures<float64_t>* features = new CDenseFeatures<float64_t>(X);

	// Create a DBN
	CDeepBeliefNetwork* dbn = new CDeepBeliefNetwork(num_features, RBMVUT_GAUSSIAN);
	dbn->add_hidden_layer(10);
	dbn->add_hidden_layer(10);
	dbn->add_hidden_layer(20);

	dbn->initialize_neural_network();

	// uncomment this line to enable info logging
	// dbn->io->set_loglevel(MSG_INFO);

	// pre-train
	dbn->pt_max_num_epochs.set_const(100);
	dbn->pt_cd_num_steps.set_const(10);
	dbn->pt_gd_learning_rate.set_const(0.01);
	dbn->pre_train(features);

	// fine-tune
	dbn->max_num_epochs = 100;
	dbn->cd_num_steps = 10;
	dbn->gd_learning_rate = 0.01;
	dbn->train(features);

	// draw 1000 samples from the DBN
	CDenseFeatures<float64_t>* samples = dbn->sample(100,1000);
	SGMatrix<float64_t> samples_matrix = samples->get_feature_matrix();

	// compute the sample means
	SGVector<float64_t> samples_means = CStatistics::matrix_mean(samples_matrix, false);

	// compute the average difference between the sample means and the true means
	float64_t avg_diff = 0;
	for (int32_t i=0; i<num_features; i++)
		avg_diff += CMath::abs(means[i]-samples_means[i]);
	avg_diff /= num_features;

	SG_SINFO("Average difference = %f\n", avg_diff);

	// Clean up
	SG_UNREF(dbn);
	SG_UNREF(features);
	SG_UNREF(samples);

	exit_shogun();
	return 0;
}
#else
int main(int, char*[])
{
	return 0;
}
#endif
