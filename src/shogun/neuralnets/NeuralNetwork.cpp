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
#include <shogun/optimization/lbfgs/lbfgs.h>

using namespace shogun;

CNeuralNetwork::CNeuralNetwork()
: CMachine()
{
	init();
}

void CNeuralNetwork::initialize(int32_t num_inputs, 
		CDynamicObjectArray* layers,
		float64_t sigma)
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
				get_layer_param_regularizable(i), sigma);
		
		get_layer(i)->set_batch_size(m_batch_size);
	}
}

CNeuralNetwork::~CNeuralNetwork()
{	
	SG_UNREF(m_layers);
}

CBinaryLabels* CNeuralNetwork::apply_binary(CFeatures* data)
{
	float64_t* output_activations = forward_propagate(data);
	SGVector<float64_t> labels_vec(m_batch_size);
	
	for (int32_t i=0; i<m_batch_size; i++)
	{
		if (get_num_outputs()==1)
		{
			if (output_activations[i]>0.5) labels_vec[i] = 1;
			else labels_vec[i] = -1;
		}
		else if (get_num_outputs()==2)
		{
			if (output_activations[2*i]>output_activations[2*i+1])
				labels_vec[i] = 1;
			else labels_vec[i] = -1;
		}
	}
	
	return new CBinaryLabels(labels_vec);
}

CRegressionLabels* CNeuralNetwork::apply_regression(CFeatures* data)
{
	float64_t* output_activations = forward_propagate(data);
	SGVector<float64_t> labels_vec(m_batch_size);
	
	for (int32_t i=0; i<m_batch_size; i++)
			labels_vec[i] = output_activations[i];
	
	return new CRegressionLabels(labels_vec);
}


CMulticlassLabels* CNeuralNetwork::apply_multiclass(CFeatures* data)
{
	float64_t* output_activations = forward_propagate(data);
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
	ASSERT(max_num_epochs>=0);
	
	float64_t* inputs = features_to_raw(data);
	float64_t* targets = labels_to_raw(m_labels);
	
	int32_t training_set_size = data->get_num_vectors();
	
	bool result = false;
	if (optimization_method==NNOM_GRADIENT_DESCENT)
		result = train_gradient_descent(inputs, targets, training_set_size);
	else if (optimization_method==NNOM_LBFGS)
		result = train_lbfgs(inputs, targets, training_set_size);
	
	SG_FREE(targets);
	return result;
}

bool CNeuralNetwork::train_gradient_descent(float64_t* inputs, 
		float64_t* targets, 
		int32_t training_set_size)
{
	ASSERT(gd_learning_rate>0);
	ASSERT(gd_momentum>=0);
	
	if (gd_mini_batch_size==0) gd_mini_batch_size = training_set_size;
	set_batch_size(gd_mini_batch_size);
	
	bool full_batch = (training_set_size==gd_mini_batch_size);
	
	int32_t n_param = get_num_parameters();
	
	// needed for momentum
	float64_t* param_updates = SG_CALLOC(float64_t, n_param);
	
	float64_t error_last_time = -1.0, error = 0;
	
	for (int32_t i=0; true; i++)
	{
		if (max_num_epochs!=0)
			if (i>max_num_epochs) break;
			
		for (int32_t j=0; j < training_set_size; j += gd_mini_batch_size)
		{
			if (j+gd_mini_batch_size>training_set_size) 
				j = training_set_size-gd_mini_batch_size;
			
			float64_t* targets_batch = targets+ j*get_num_outputs();
			float64_t* inputs_batch = inputs + j*m_num_inputs;
			
			compute_gradients(inputs_batch, targets_batch);
			error = compute_error(targets_batch);
			
			for (int32_t k=0; k<n_param; k++)
			{
				param_updates[k] = gd_momentum*param_updates[k]
						-gd_learning_rate*m_param_gradients[k];
					
				m_params[k] += param_updates[k];
			}
			
			if (print_during_training)
				SG_SPRINT("Epoch %i: Error = %f\n",i, error);
		}
		if (full_batch && error_last_time!=-1.0)
			if (CMath::abs(error_last_time-error)/error < epsilon) break;
		
		error_last_time = error;
	}
	
	SG_FREE(param_updates);
	
	return true;
}

bool CNeuralNetwork::train_lbfgs(float64_t* inputs, 
		float64_t* targets, 
		int32_t training_set_size)
{
	set_batch_size(training_set_size);
	
	lbfgs_parameter_t lbfgs_param;
	lbfgs_parameter_init(&lbfgs_param);
	lbfgs_param.max_iterations = max_num_epochs;
	lbfgs_param.epsilon = 0;
	lbfgs_param.past = 1;
	lbfgs_param.delta = epsilon;
	
	m_lbfgs_temp_inputs = inputs;
	m_lbfgs_temp_targets = targets;

	int32_t result = lbfgs(m_total_num_parameters, 
			m_params, 
			NULL, 
			&CNeuralNetwork::lbfgs_evaluate, 
			&CNeuralNetwork::lbfgs_progress, 
			this, 
			&lbfgs_param);
	
	m_lbfgs_temp_inputs = NULL;
	m_lbfgs_temp_targets = NULL;
	
	if (result==LBFGS_SUCCESS || 1) 
	{
		if (print_during_training) SG_SPRINT("L-BFGS Optimization Converged\n");
	}
	else if (result==LBFGSERR_MAXIMUMITERATION)
	{
		if (print_during_training) 
			SG_SPRINT("L-BFGS Max Number of Epochs reached\n");
	}
	else
	{
		if (print_during_training) 
			SG_SPRINT("L-BFGS optimization ended with return code %i\n",result);
	}
	return true;
}

float64_t CNeuralNetwork::lbfgs_evaluate(void* userdata, 
		const float64_t* W, 
		float64_t* grad, 
		const int32_t n, 
		const float64_t step)
{
	CNeuralNetwork* network = static_cast<CNeuralNetwork*>(userdata);
	
	network->compute_gradients(network->m_lbfgs_temp_inputs, 
			network->m_lbfgs_temp_targets, 
			grad);
	
	return network->compute_error(network->m_lbfgs_temp_targets);
}

int CNeuralNetwork::lbfgs_progress(void* instance, 
		const float64_t* x, 
		const float64_t* g, 
		const float64_t fx, 
		const float64_t xnorm, 
		const float64_t gnorm, 
		const float64_t step, 
		int n, int k, int ls)
{
	CNeuralNetwork* network = static_cast<CNeuralNetwork*>(instance);
	if (network->print_during_training) 
		SG_SPRINT("Epoch %i: Error = %f\n",k, fx);
	return 0;
}


float64_t* CNeuralNetwork::forward_propagate(CFeatures* data)
{
	float64_t* inputs = features_to_raw(data);
	set_batch_size(data->get_num_vectors());	
	return forward_propagate(inputs);
}


float64_t* CNeuralNetwork::forward_propagate(float64_t* inputs)
{
	// forward propagation
	get_layer(0)->compute_activations(get_layer_params(0), inputs);
	
	for (int i=1; i<m_num_layers; i++)
		get_layer(i)->compute_activations(get_layer_params(i),
				get_layer(i-1)->get_activations());
	
	return get_layer(m_num_layers-1)->get_activations();
}

void CNeuralNetwork::compute_gradients(float64_t* inputs, 
		float64_t* targets,
		float64_t* gradients)
{
	float64_t* param_gradients_backup = m_param_gradients.vector;
	if (gradients!=NULL) m_param_gradients.vector = gradients;
	
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
	if (l2_coefficient != 0.0)
	{
		for (int32_t i=0; i<m_total_num_parameters; i++)
		{
			if (m_param_regularizable[i])
				m_param_gradients[i] += l2_coefficient*m_params[i];
		}
	}
	
	m_param_gradients.vector = param_gradients_backup;
}

float64_t CNeuralNetwork::compute_error(float64_t* targets, float64_t* inputs)
{
	if (inputs!=NULL) forward_propagate(inputs);
	
	float64_t error = get_layer(m_num_layers-1)->computer_error(targets);
	
	// L2 regularization
	if (l2_coefficient != 0.0)
	{
		for (int32_t i=0; i<m_total_num_parameters; i++)
		{
			if (m_param_regularizable[i]) 
				error += 0.5*l2_coefficient*m_params[i]*m_params[i];
		}
	}
	
	return error;
}

bool CNeuralNetwork::check_gradients(float64_t approx_epsilon, float64_t tolerance)
{
	// some random inputs and ouputs
	SGVector<float64_t> x(m_num_inputs);
	SGVector<float64_t> y(get_num_outputs());
	x.random(0.0, 1.0);
	
	// the outputs are set up in the form of a probability distribution (in case
	// that is required by the output layer, i.e softmax)
	y.random(0.0, 1.0);
	float64_t y_sum = SGVector<float64_t>::sum(y.vector, y.vlen);
	for (int32_t i=0; i<y.vlen; i++) y[i] /= y_sum;
	
	set_batch_size(1);
	
	// numerically compute gradients
	float64_t* gradients_numerical =SG_MALLOC(float64_t,m_total_num_parameters);
	for (int32_t i=0; i<m_total_num_parameters; i++)
	{
		m_params[i] += approx_epsilon;
		float64_t error_plus = compute_error(y.vector, x.vector);
		m_params[i] -= 2*approx_epsilon;
		float64_t error_minus = compute_error(y.vector, x.vector);
		m_params[i] += approx_epsilon;
		
		gradients_numerical[i] = (error_plus-error_minus)/(2*approx_epsilon);
	}
	
	// compute gradients using backpropagation
	compute_gradients(x.vector, y.vector);
	
	// compare
	for (int32_t i=0; i<m_total_num_parameters; i++)
	{
		float64_t diff = m_param_gradients[i] - gradients_numerical[i];
		
		if (CMath::abs(diff) > tolerance) return false;
	}
	
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

float64_t* CNeuralNetwork::labels_to_raw(CLabels* labs)
{
	float64_t* targets = SG_CALLOC(float64_t, 
		get_num_outputs()*labs->get_num_labels());
	
	if (labs->get_label_type() == LT_MULTICLASS)
	{
		CMulticlassLabels* labels_mc = (CMulticlassLabels*) labs;
		ASSERT(labels_mc->get_num_classes()==get_num_outputs());
		
		for (int32_t i=0; i<labels_mc->get_num_labels(); i++)
			targets[((int32_t)labels_mc->get_label(i))+ i*get_num_outputs()] 
				= 1.0;
	}
	else if (labs->get_label_type() == LT_BINARY)
	{
		CBinaryLabels* labels_bin = (CBinaryLabels*) labs;
		if (get_num_outputs()==1)
		{
			for (int32_t i=0; i<labels_bin->get_num_labels(); i++)
				targets[i] = (labels_bin->get_label(i)==1);
		}
		else if (get_num_outputs()==2)
		{
			for (int32_t i=0; i<labels_bin->get_num_labels(); i++)
			{
				targets[i*2] = (labels_bin->get_label(i)==1);
				targets[i*2+1] = (labels_bin->get_label(i)==-1);
			}
		}
	}
	else if (labs->get_label_type() == LT_REGRESSION)
	{
		CRegressionLabels* labels_reg = (CRegressionLabels*) labs;
		for (int32_t i=0; i<labels_reg->get_num_labels(); i++)
			targets[i] = labels_reg->get_label(i);
	}
	return targets;
}


EProblemType CNeuralNetwork::get_machine_problem_type() const
{
	// problem type depends on the type of labels given to the network
	// if no labels are given yet, just return PT_MULTICLASS
	if (m_labels==NULL) 
		return PT_MULTICLASS;
	
	if (m_labels->get_label_type() == LT_BINARY)
		return PT_BINARY;
	else if (m_labels->get_label_type() == LT_REGRESSION)
		return PT_REGRESSION;
	else return PT_MULTICLASS;
}

bool CNeuralNetwork::is_label_valid(CLabels* lab) const
{
	return (lab->get_label_type() == LT_MULTICLASS ||
		lab->get_label_type() == LT_BINARY ||
		lab->get_label_type() == LT_REGRESSION);
}

void CNeuralNetwork::set_labels(CLabels* lab)
{
	if (lab->get_label_type() == LT_BINARY)
	{
		if (get_num_outputs() > 2)
		{
			SG_ERROR("Cannot use %s in a neural network with more that 2"	
				" output neurons", lab->get_name());
			return;
		}	
	}
	else if (lab->get_label_type() == LT_REGRESSION)
	{
		if (get_num_outputs() > 1)
		{
			SG_ERROR("Cannot use %s in a neural network with more that 1"	
				" output neuron", lab->get_name());
			return;
		}
	}
	shogun::CMachine::set_labels(lab);
}


void CNeuralNetwork::init()
{
	optimization_method = NNOM_LBFGS;
	print_during_training = true;
	l2_coefficient = 0.0; 
	gd_mini_batch_size = 0;
	max_num_epochs = 0; 
	gd_learning_rate = 0.1; 
	gd_momentum = 0.9; 
	epsilon = 1.0e-5;
	m_num_inputs = 0;
	m_num_layers = 0;
	m_layers = NULL;
	m_total_num_parameters = 0;
	m_batch_size = 1;
	m_lbfgs_temp_inputs = NULL;
	m_lbfgs_temp_targets = NULL;
	
	SG_ADD((machine_int_t*)&optimization_method, "optimization_method",
	       "Optimization Method", MS_NOT_AVAILABLE);
	SG_ADD(&print_during_training, "print_during_training",
	       "Print During Training", MS_NOT_AVAILABLE);
	SG_ADD(&gd_mini_batch_size, "gd_mini_batch_size",
	       "Gradient Descent Mini-batch size", MS_NOT_AVAILABLE);
	SG_ADD(&max_num_epochs, "max_num_epochs",
	       "Max number of Epochs", MS_NOT_AVAILABLE);
	SG_ADD(&gd_learning_rate, "gd_learning_rate",
	       "Gradient descent learning rate", MS_NOT_AVAILABLE);
	SG_ADD(&gd_momentum, "gd_momentum",
	       "Gradient Descent Momentum", MS_NOT_AVAILABLE);
	SG_ADD(&epsilon, "epsilon",
	       "Epsilon", MS_NOT_AVAILABLE);
	SG_ADD(&m_num_inputs, "num_inputs",
	       "Number of Inputs", MS_NOT_AVAILABLE);
	SG_ADD(&m_num_layers, "num_layers",
	       "Number of Layers", MS_NOT_AVAILABLE);
	SG_ADD(&l2_coefficient, "l2_coefficient",
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
