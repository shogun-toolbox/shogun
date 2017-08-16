/*
 * Copyright (c) 2017, Shogun Toolbox Foundation
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
 * Written (W) 2017 Olivier Nguyen
 */

#include <shogun/neuralnets/NeuralRecurrentNetwork.h>
#include <shogun/mathematics/Math.h>
#include <shogun/optimization/lbfgs/lbfgs.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/lib/DynamicObjectArray.h>
#include <shogun/neuralnets/NeuralLayer.h>

using namespace shogun;

CNeuralRecurrentNetwork::CNeuralRecurrentNetwork()
: CNeuralNetwork()
{
	init();
}

CNeuralRecurrentNetwork::CNeuralRecurrentNetwork(CDynamicObjectArray* layers)
{
	init();
	set_layers(layers);
}

void CNeuralRecurrentNetwork::initialize_neural_network(float64_t sigma)
{
}

CNeuralRecurrentNetwork::~CNeuralRecurrentNetwork()
{
	SG_UNREF(m_layers);
}

CBinaryLabels* CNeuralRecurrentNetwork::apply_binary(CFeatures* data)
{
	SGMatrix<float64_t> output_activations = forward_propagate(data);
	CBinaryLabels* labels = new CBinaryLabels(m_batch_size);

	return labels;
}

CRegressionLabels* CNeuralRecurrentNetwork::apply_regression(CFeatures* data)
{
	SGMatrix<float64_t> output_activations = forward_propagate(data);
	SGVector<float64_t> labels_vec(m_batch_size);

	return new CRegressionLabels(labels_vec);
}


CMulticlassLabels* CNeuralRecurrentNetwork::apply_multiclass(CFeatures* data)
{
	SGMatrix<float64_t> output_activations = forward_propagate(data);
	SGVector<float64_t> labels_vec(m_batch_size);

	CMulticlassLabels* labels = new CMulticlassLabels(labels_vec);

	return labels;
}

bool CNeuralRecurrentNetwork::train_machine(CFeatures* data)
{
	return true;
}

bool CNeuralRecurrentNetwork::train_gradient_descent(SGMatrix<float64_t> inputs,
		SGMatrix<float64_t> targets)
{
	return true;
}

SGMatrix<float64_t> CNeuralRecurrentNetwork::forward_propagate(CFeatures* data, int32_t j)
{
	SGMatrix<float64_t> inputs = features_to_matrix(data);
	set_batch_size(data->get_num_vectors());
	return forward_propagate(inputs, j);
}

SGMatrix<float64_t> CNeuralRecurrentNetwork::forward_propagate(
	SGMatrix<float64_t> inputs, int32_t j)
{
	return 0.0;
}

float64_t CNeuralRecurrentNetwork::compute_gradients(SGMatrix<float64_t> inputs,
		SGMatrix<float64_t> targets, SGVector<float64_t> gradients)
{
	return 0.0;
}

float64_t CNeuralRecurrentNetwork::compute_error(SGMatrix<float64_t> targets)
{
	return 0.0;
}

float64_t CNeuralRecurrentNetwork::compute_error(SGMatrix<float64_t> inputs,
		SGMatrix<float64_t> targets)
{
	forward_propagate(inputs);
	return compute_error(targets);
}

void CNeuralRecurrentNetwork::init()
{
}
