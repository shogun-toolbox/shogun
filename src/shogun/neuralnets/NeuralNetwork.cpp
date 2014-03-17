/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 Khaled Nasr
 */

#include <shogun/mathematics/Math.h>
#include <shogun/base/Parameter.h>
#include <shogun/neuralnets/NeuralNetwork.h>


using namespace shogun;


CNeuralNetwork::CNeuralNetwork()
: CSGObject(), m_num_inputs(0), m_num_layers(0), m_layers(NULL),
	m_L2_coeff(0.0), m_total_num_parameters(), m_batch_size(1)
	
{
	init();
}

CNeuralNetwork::CNeuralNetwork(const CNeuralNetwork& orig) :CSGObject()
{
	shallow_copy(orig);
	init();
}


void CNeuralNetwork::initialize(int32_t num_inputs, CDynamicObjectArray* layers)
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

CDenseFeatures<float64_t>* CNeuralNetwork::apply(
		CDenseFeatures<float64_t>* inputs)
{
	ASSERT(inputs->get_num_features()==m_num_inputs);
	
	set_batch_size(inputs->get_num_vectors());
	
	float64_t* inputs_matrix;
	int32_t dummy;
	inputs_matrix = inputs->get_feature_matrix(dummy, dummy);
	
	forward_propagate(inputs_matrix);
	
	SGMatrix<float64_t> result(get_num_outputs(), m_batch_size);
	
	int32_t length = get_num_outputs()*m_batch_size;
	
	float64_t* output_activations =get_layer(m_num_layers-1)->get_activations();
	
	for (int32_t i=0; i<length; i++)
		result.matrix[i] = output_activations[i];

	return new CDenseFeatures<float64_t>(result);
}

void CNeuralNetwork::train_gradient_descent(
		CDenseFeatures< float64_t >* inputs,	
		CDenseFeatures< float64_t >* targets,
		int32_t max_num_epochs,
		int32_t batch_size, 
		float64_t learning_rate,		
		float64_t momentum)
{
	int32_t training_set_size = inputs->get_num_vectors();
	int32_t _batch_size = batch_size;
	if (_batch_size==0) _batch_size = training_set_size;
	
	ASSERT(training_set_size >= _batch_size);
	ASSERT(inputs->get_num_features()==m_num_inputs);
	ASSERT(targets->get_num_features()==get_num_outputs());
	ASSERT(targets->get_num_vectors()==inputs->get_num_vectors());
	
	ASSERT(learning_rate>0);
	ASSERT(max_num_epochs>0);
	ASSERT(momentum>=0);
	
	float64_t* inputs_matrix; 
	float64_t* targets_matrix;
	int32_t dummy;
	inputs_matrix = inputs->get_feature_matrix(dummy, dummy);
	targets_matrix = targets->get_feature_matrix(dummy, dummy);
	
	int32_t n_param = get_num_parameters();
	
	// needed for momentum
	float64_t* param_updates = SG_MALLOC(float64_t, n_param);
	for (int32_t i=0; i<n_param; i++) param_updates[i] = 0.0;
	
	set_batch_size(_batch_size);	
	
	for (int32_t i=0; i<max_num_epochs; i++)
	{
		for (int32_t j=0; j < training_set_size; j += _batch_size)
		{
			if (j+_batch_size>training_set_size) j = 
												training_set_size-_batch_size;
			
			float64_t* targets_batch = targets_matrix + j*get_num_outputs();
			float64_t* inputs_batch = inputs_matrix + j*m_num_inputs;
			
			compute_gradients(inputs_batch, targets_batch);
			float64_t error = compute_error(targets_batch);
			
			for (int32_t k=0; k<n_param; k++)
			{
				param_updates[k] = momentum*param_updates[k]
									-learning_rate*m_param_gradients[k];
					
				m_params[k] += param_updates[k];
			}
			
			SG_SPRINT("Epoch %i: Error = %f\n",i, error);
		}
	}
	
	SG_FREE(param_updates);
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

void CNeuralNetwork::init()
{
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
			"DynamicObjectArray of NeuralNetwork objects", MS_NOT_AVAILABLE);
}

void CNeuralNetwork::shallow_copy(const CNeuralNetwork& orig)
{
	m_num_inputs = orig.m_num_inputs;
	m_L2_coeff = orig.m_L2_coeff;
	m_layers = orig.m_layers;
	m_num_inputs = orig.m_num_inputs;
	m_num_layers = orig.m_num_layers;
	m_total_num_parameters = orig.m_total_num_parameters;
	m_batch_size = orig.m_batch_size;
	m_params = SGVector<float64_t>(orig.m_params);
	m_param_gradients = SGVector<float64_t>(orig.m_param_gradients);
	m_param_regularizable = SGVector<bool>(orig.m_param_regularizable);
}
