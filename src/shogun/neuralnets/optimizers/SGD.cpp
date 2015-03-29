/*
 * Copyright (c) 2015, Shogun Toolbox Foundation
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
 * Written (W) 2015 Khaled Nasr, Sergey Lisitsyn
 */

#include <shogun/neuralnets/optimizers/SGD.h> 
#include <shogun/lib/SGMatrix.h>
#include <shogun/base/SGObject.h>
#include <shogun/io/SGIO.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/neuralnets/NeuralNetwork.h>

namespace shogun
{

CSGDNeuralNetworkOptimizer::CSGDNeuralNetworkOptimizer() :
	CNeuralNetworkOptimizer()
{
	prepare();
}

void CSGDNeuralNetworkOptimizer::prepare()
{
	gd_mini_batch_size = 0;
	gd_learning_rate = 0.1;
	gd_learning_rate_decay = 1.0;
	gd_momentum = 0.9;
	gd_error_damping_coeff = -1;

	SG_ADD(&gd_mini_batch_size, "gd_mini_batch_size",
			"Mini-batch size", MS_NOT_AVAILABLE);
	SG_ADD(&gd_learning_rate, "gd_learning_rate", 
			"Learning rate", MS_NOT_AVAILABLE);
	SG_ADD(&gd_learning_rate_decay, "gd_learning_rate_decay", 
			"Decay of learning rate", MS_NOT_AVAILABLE);
	SG_ADD(&gd_momentum, "gd_momentum",
			"Momentum of updates", MS_NOT_AVAILABLE);
	SG_ADD(&gd_error_damping_coeff, "gd_error_damping_coeff",
			"Error damping coefficient", MS_NOT_AVAILABLE);
}

bool CSGDNeuralNetworkOptimizer::optimize(CNeuralNetwork* network,
		CDotFeatures* inputs,
		SGMatrix<float64_t> targets)
{
	REQUIRE(gd_learning_rate>0,
		"Gradient descent learning rate (%f) must be > 0\n", gd_learning_rate);
	REQUIRE(gd_momentum>=0,
		"Gradient descent momentum (%f) must be > 0\n", gd_momentum);

	int32_t training_set_size = inputs->get_num_vectors();
	if (gd_mini_batch_size==0) gd_mini_batch_size = training_set_size;
	network->set_batch_size(gd_mini_batch_size);

	int32_t n_param = network->get_num_parameters();
	SGVector<float64_t> gradients(n_param);

	// needed for momentum
	SGVector<float64_t> param_updates(n_param);
	param_updates.zero();

	float64_t error_last_time = -1.0, error = -1.0;

	float64_t c = gd_error_damping_coeff;
	if (c==-1.0)
		c = 0.99*(float64_t)gd_mini_batch_size/training_set_size + 1e-2;

	bool continue_training = true;
	float64_t alpha = gd_learning_rate;
	SGVector<index_t> inputs_batch_subset(gd_mini_batch_size);
	SGMatrix<float64_t> targets_batch;

	for (int32_t i=0; continue_training; i++)
	{
		if (network->max_num_epochs!=0)
			if (i>=network->max_num_epochs) break;

		for (int32_t j=0; j < training_set_size; j += gd_mini_batch_size)
		{
			alpha = gd_learning_rate_decay*alpha;

			if (j+gd_mini_batch_size>training_set_size)
				j = training_set_size-gd_mini_batch_size;


			targets_batch = 
				SGMatrix<float64_t>(targets.matrix+j*network->get_num_outputs(),
					network->get_num_outputs(), gd_mini_batch_size, false);

			for (int32_t k=0; k<n_param; k++)
				network->get_params()[k] += gd_momentum*param_updates[k];

			for (int32_t k=j; k<gd_mini_batch_size; k++)
				inputs_batch_subset[k-j] = k;

			inputs->add_subset(inputs_batch_subset);
			float64_t e = 
				network->compute_gradients(inputs, targets_batch, gradients);
			inputs->remove_subset();

			// filter the errors
			if (error==-1.0)
				error = e;
			else
				error = (1.0-c) * error + c*e;

			for (int32_t k=0; k<n_param; k++)
			{
				param_updates[k] = gd_momentum*param_updates[k]
						-alpha*gradients[k];

				network->get_params()[k] -= alpha*gradients[k];
			}

			if (error_last_time!=-1.0)
			{
				float64_t error_change = (error_last_time-error)/error;
				if (error_change< network->epsilon && error_change>=0)
				{
					SG_INFO("Gradient Descent Optimization Converged\n");
					continue_training = false;
					break;
				}

				SG_INFO("Epoch %i: Error = %f\n",i, error);
			}
			error_last_time = error;
		}
	}

	return true;
}

}
