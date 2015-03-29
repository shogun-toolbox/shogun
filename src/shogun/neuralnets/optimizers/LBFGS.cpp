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

#include <shogun/neuralnets/optimizers/LBFGS.h> 
#include <shogun/neuralnets/NeuralNetwork.h>
#include <shogun/optimization/lbfgs/lbfgs.h>

namespace shogun
{

struct CNeuralNetworkTrainingContext
{
	CNeuralNetwork* network;
	CDotFeatures* features;
	SGMatrix<float64_t> targets;
};

CLBFGSNeuralNetworkOptimizer::CLBFGSNeuralNetworkOptimizer() :
	CNeuralNetworkOptimizer()
{
}

bool CLBFGSNeuralNetworkOptimizer::optimize(CNeuralNetwork* network,
		CDotFeatures* input, const SGMatrix<float64_t> targets)
{
	int32_t training_set_size = input->get_num_vectors();
	network->set_batch_size(training_set_size);

	lbfgs_parameter_t lbfgs_param;
	lbfgs_parameter_init(&lbfgs_param);
	lbfgs_param.max_iterations = network->max_num_epochs;
	lbfgs_param.epsilon = 0;
	lbfgs_param.past = 1;
	lbfgs_param.delta = network->epsilon;

	CNeuralNetworkTrainingContext context;
	context.network = network;
	context.features = input;
	context.targets = targets;

	int32_t result = lbfgs(network->get_total_num_parameters(),
			network->get_params(),
			NULL,
			&CLBFGSNeuralNetworkOptimizer::lbfgs_evaluate,
			&CLBFGSNeuralNetworkOptimizer::lbfgs_progress,
			&context,
			&lbfgs_param);

	if (result==LBFGS_SUCCESS || 1)
	{
		SG_INFO("L-BFGS Optimization Converged\n");
	}
	else if (result==LBFGSERR_MAXIMUMITERATION)
	{
		SG_INFO("L-BFGS Max Number of Epochs reached\n");
	}
	else
	{
		SG_INFO("L-BFGS optimization ended with return code %i\n",result);
	}
	return true;
}

float64_t CLBFGSNeuralNetworkOptimizer::lbfgs_evaluate(void* userdata,
		const float64_t* W,
		float64_t* grad,
		const int32_t n,
		const float64_t step)
{
	CNeuralNetworkTrainingContext* context =
		static_cast<CNeuralNetworkTrainingContext*>(userdata);
	CNeuralNetwork* network = context->network;
	CDotFeatures* features = context->features;
	SGMatrix<float64_t> targets = context->targets;

	SGVector<float64_t> grad_vector(grad, network->get_num_parameters(), false);

	return network->compute_gradients(features,
		targets, grad_vector);
}

int CLBFGSNeuralNetworkOptimizer::lbfgs_progress(void* instance,
		const float64_t* x,
		const float64_t* g,
		const float64_t fx,
		const float64_t xnorm,
		const float64_t gnorm,
		const float64_t step,
		int n, int k, int ls)
{
	SG_SINFO("Epoch %i: Error = %f\n",k, fx);
	return 0;
}


}
