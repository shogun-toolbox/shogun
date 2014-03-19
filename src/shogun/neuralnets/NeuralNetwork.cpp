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

#include <shogun/mathematics/Math.h>
#include <shogun/base/Parameter.h>
#include <shogun/neuralnets/NeuralNetwork.h>

using namespace shogun;

CNeuralNetwork::CNeuralNetwork()
: CMachine(), mini_batch_size(0), max_num_epochs(100), 
	gd_learning_rate(0.1), gd_momentum(0.9), m_num_inputs(0), 
	m_num_layers(0), m_layers(NULL), m_L2_coeff(0.0), 
	m_total_num_parameters(), m_batch_size(1)
{
	init();
}

void CNeuralNetwork::initialize(int32_t num_inputs, 
		CDynamicObjectArray* layers)
{
	m_num_inputs = num_inputs;
	m_num_layers = layers->get_num_elements();
	m_layers = layers;
	
	SG_REF(m_layers);
	
	get_layer(0)->initialize(m_num_inputs);
	for (int32_t i=1; i<m_num_layers; i++) 
		get_layer(i)->initialize(get_layer(i-1)->get_num_neurons());
	
	m_index_offsets = SGVector<int32_t>(m_num_layers);
	
	m_total_num_parameters = get_layer(0)->get_num_parameters();
	m_index_offsets[0] = 0;
	for (int32_t i=1; i<m_num_layers; i++)
	{
		m_index_offsets[i] = m_total_num_parameters;
		m_total_num_parameters += get_layer(i)->get_num_parameters();
	}
	
	m_params = SGVector<float64_t>(m_total_num_parameters);
	m_param_gradients = SGVector<float64_t>(m_total_num_parameters);
	m_param_regularizable = SGVector<bool>(m_total_num_parameters);
	
	m_params.zero();
	m_param_gradients.zero();
	m_param_regularizable.set_const(true);
	
	for (int32_t i=0; i<m_num_layers; i++)
	{
		get_layer(i)->initialize_parameters(get_layer_params(i),	
				get_layer_param_regularizable(i));
		
		get_layer(i)->set_batch_size(m_batch_size);
	}
}

CNeuralNetwork::~CNeuralNetwork()
{	
	SG_UNREF(m_layers);
}

CMulticlassLabels* CNeuralNetwork::apply_multiclass(CFeatures* data)
{
	float64_t* inputs = features_to_raw(data);
	set_batch_size(data->get_num_vectors());	
	forward_propagate(inputs);
	
	float64_t* output_activations =get_layer(m_num_layers-1)->get_activations();
	SGVector<float64_t> labels_vec(m_batch_size);
	
	for (int32_t i=0; i<m_batch_size; i++)
	{
		labels_vec[i] = SGVector<float64_t>::arg_max(
			output_activations+i*get_num_outputs(), 1, get_num_outputs());
	}
	
	return new CMulticlassLabels(labels_vec);
}

bool CNeuralNetwork::train_machine(CFeatures* data)
{
	float64_t* inputs = features_to_raw(data);
	
	CMulticlassLabels* labels = (CMulticlassLabels*) m_labels;
	
	ASSERT(labels->get_num_classes()==get_num_outputs());
	ASSERT(labels->get_num_labels()==data->get_num_vectors());
	
	ASSERT(max_num_epochs>0);
	ASSERT(gd_learning_rate>0);
	ASSERT(gd_momentum>=0);
	
	int32_t training_set_size = data->get_num_vectors();
	if (mini_batch_size==0) mini_batch_size = training_set_size;
	set_batch_size(mini_batch_size);
	
	SGMatrix<float64_t> targets_matrix(get_num_outputs(), training_set_size);
	targets_matrix.zero();
	
	for (int32_t i=0; i<training_set_size; i++)
		targets_matrix((int32_t)labels->get_label(i), i) = 1.0;

	float64_t* targets = targets_matrix.matrix;
	
	int32_t n_param = get_num_parameters();
	
	// needed for momentum
	float64_t* param_updates = SG_CALLOC(float64_t, n_param);
		
	for (int32_t i=0; i<max_num_epochs; i++)
	{
		for (int32_t j=0; j < training_set_size; j += mini_batch_size)
		{
			if (j+mini_batch_size>training_set_size) 
				j = training_set_size-mini_batch_size;
			
			float64_t* targets_batch = targets+ j*get_num_outputs();
			float64_t* inputs_batch = inputs + j*m_num_inputs;
			
			compute_gradients(inputs_batch, targets_batch);
			float64_t error = compute_error(targets_batch);
			
			for (int32_t k=0; k<n_param; k++)
			{
				param_updates[k] = gd_momentum*param_updates[k]
						-gd_learning_rate*m_param_gradients[k];
					
				m_params[k] += param_updates[k];
			}
			
			SG_SPRINT("Epoch %i: Error = %f\n",i, error);
		}
	}
	
	SG_FREE(param_updates);
	
	return true;
}

void CNeuralNetwork::forward_propagate(float64_t* inputs)
{
	// forward propagation
	get_layer(0)->compute_activations(get_layer_params(0), inputs);
	
	for (int i=1; i<m_num_layers; i++)
		get_layer(i)->compute_activations(get_layer_params(i),
				get_layer(i-1)->get_activations());
}

void CNeuralNetwork::compute_gradients(float64_t* inputs, float64_t* targets)
{
	forward_propagate(inputs);

	if (m_num_layers==1)
	{
		get_layer(0)->compute_gradients(get_layer_params(0), true, targets, 
				inputs ,get_layer_param_gradients(0));
	}
	else
	{
		// backpropagation
		for (int32_t i=m_num_layers-1; i>=0; i--)
		{
			if (i==m_num_layers-1)
			{
				get_layer(i)->compute_gradients(get_layer_params(i), true, 
						targets, get_layer(i-1)->get_activations(),
						get_layer_param_gradients(i));
			}
			else if (i==0)
			{
				get_layer(i)->compute_gradients(get_layer_params(i), false,
						get_layer(i+1)->get_input_gradients(),
						inputs, get_layer_param_gradients(i));
			}
			else 
			{
				get_layer(i)->compute_gradients(get_layer_params(i), false,
						get_layer(i+1)->get_input_gradients(),
						get_layer(i-1)->get_activations(),
						get_layer_param_gradients(i));
			}
		}
		
	}
	
	// L2 regularization
	if (m_L2_coeff != 0.0)
	{
		for (int32_t i=0; i<m_total_num_parameters; i++)
		{
			if (m_param_regularizable[i])
				m_param_gradients[i] += m_L2_coeff*m_params[i];
		}
	}
}

float64_t CNeuralNetwork::compute_error(float64_t* targets, float64_t* inputs)
{
	if (inputs!=NULL) forward_propagate(inputs);
	
	return get_layer(m_num_layers-1)->computer_error(targets);
}

bool CNeuralNetwork::check_gradients(float64_t epsilon, float64_t tolerance)
{
	// some random inputs and ouputs
	SGVector<float64_t> x(m_num_inputs);
	SGVector<float64_t> y(get_num_outputs());
	x.random(0.0, 1.0);
	y.random(0.0, 1.0);
	set_batch_size(1);
	
	// disable regularization
	float64_t L2_coeff = m_L2_coeff;
	set_L2_regularization(0.0);
	
	// numerically compute gradients
	float64_t* gradients_numerical =SG_MALLOC(float64_t,m_total_num_parameters);
	for (int32_t i=0; i<m_total_num_parameters; i++)
	{
		m_params[i] += epsilon;
		float64_t error_plus = compute_error(y.vector, x.vector);
		m_params[i] -= 2*epsilon;
		float64_t error_minus = compute_error(y.vector, x.vector);
		m_params[i] += epsilon;
		
		gradients_numerical[i] = (error_plus-error_minus)/(2*epsilon);
	}
	
	// compute gradients using backpropagation
	compute_gradients(x.vector, y.vector);
	
	// compare
	for (int32_t i=0; i<m_total_num_parameters; i++)
	{
		float64_t diff = m_param_gradients[i] - gradients_numerical[i];
		
		if (CMath::abs(diff) > tolerance) return false;
	}
	
	// restore regularization parameter
	set_L2_regularization(L2_coeff);
	
	SG_FREE(gradients_numerical);
	return true;
}


void CNeuralNetwork::set_batch_size(int32_t batch_size)
{
	if (batch_size!=m_batch_size)
	{
		m_batch_size = batch_size;
		for (int32_t i=0; i<m_num_layers; i++)
			get_layer(i)->set_batch_size(m_batch_size);
	}
}

float64_t* CNeuralNetwork::features_to_raw(CFeatures* features)
{
	ASSERT(features->get_feature_type() == F_DREAL);
	ASSERT(features->get_feature_class() == C_DENSE);
	
	CDenseFeatures<float64_t>* inputs = (CDenseFeatures<float64_t>*) features;
	ASSERT(inputs->get_num_features()==m_num_inputs);
	
	int32_t dummy;
	float64_t* inputs_matrix = inputs->get_feature_matrix(dummy, dummy);
	return inputs_matrix;
}

void CNeuralNetwork::init()
{
	SG_ADD(&mini_batch_size, "mini_batch_size",
	       "Mini-batch size", MS_NOT_AVAILABLE);
	SG_ADD(&max_num_epochs, "max_num_epochs",
	       "Max number of Epochs", MS_NOT_AVAILABLE);
	SG_ADD(&gd_learning_rate, "gd_learning_rate",
	       "Gradient descent learning rate", MS_NOT_AVAILABLE);
	SG_ADD(&gd_momentum, "gd_momentum",
	       "Gradient Descent Momentum", MS_NOT_AVAILABLE);
	SG_ADD(&m_num_inputs, "num_inputs",
	       "Number of Inputs", MS_NOT_AVAILABLE);
	SG_ADD(&m_num_layers, "num_layers",
	       "Number of Layers", MS_NOT_AVAILABLE);
	SG_ADD(&m_L2_coeff, "L2_coeff",
	       "L2 regularization coeff", MS_NOT_AVAILABLE);
	SG_ADD(&m_total_num_parameters, "total_num_parameters",
	       "Total number of parameters", MS_NOT_AVAILABLE);
	SG_ADD(&m_batch_size, "batch_size",
	       "Batch Size", MS_NOT_AVAILABLE);
	SG_ADD(&m_index_offsets, "index_offsets",
		"Index Offsets", MS_NOT_AVAILABLE);
	SG_ADD(&m_params, "params",
		"Parameters", MS_NOT_AVAILABLE);
	SG_ADD(&m_param_gradients, "param_gradients",
		"Parameter Gradients", MS_NOT_AVAILABLE);
	SG_ADD(&m_param_regularizable, "param_regularizable",
		"Parameter Regularizable", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_layers, "layers", 
		"DynamicObjectArray of NeuralNetwork objects",
		MS_NOT_AVAILABLE);
}
