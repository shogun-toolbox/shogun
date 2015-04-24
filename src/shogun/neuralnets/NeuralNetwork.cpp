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

#include <shogun/neuralnets/NeuralNetwork.h>
#include <shogun/mathematics/Math.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/lib/DynamicObjectArray.h>
#include <shogun/neuralnets/NeuralLayer.h>
#include <shogun/neuralnets/optimizers/LBFGS.h>

using namespace shogun;

CNeuralNetwork::CNeuralNetwork()
: CMachine()
{
	init();
}

CNeuralNetwork::CNeuralNetwork(CDynamicObjectArray* layers)
{
	init();
	set_layers(layers);
}

void CNeuralNetwork::set_layers(CDynamicObjectArray* layers)
{
	REQUIRE(layers, "Layers should not be NULL")

	SG_UNREF(m_layers);
	SG_REF(layers);
	m_layers = layers;

	m_num_layers = m_layers->get_num_elements();
	m_adj_matrix = SGMatrix<bool>(m_num_layers, m_num_layers);
	m_adj_matrix.zero();

	m_num_inputs = 0;
	for (int32_t i=0; i<m_num_layers; i++)
	{
		if (get_layer(i)->is_input())
			m_num_inputs += get_layer(i)->get_num_neurons();
	}
}

void CNeuralNetwork::connect(int32_t i, int32_t j)
{
	REQUIRE("i<j", "i(%i) must be less that j(%i)\n", i, j);
	m_adj_matrix(i,j) = true;
}

void CNeuralNetwork::quick_connect()
{
	m_adj_matrix.zero();
	for (int32_t i=1; i<m_num_layers; i++)
		m_adj_matrix(i-1, i) = true;
}

void CNeuralNetwork::disconnect(int32_t i, int32_t j)
{
	m_adj_matrix(i,j) = false;
}

void CNeuralNetwork::disconnect_all()
{
	m_adj_matrix.zero();
}

void CNeuralNetwork::initialize(float64_t sigma)
{
	for (int32_t j=0; j<m_num_layers; j++)
	{
		if (!get_layer(j)->is_input())
		{
			int32_t num_inputs = 0;
			for (int32_t i=0; i<m_num_layers; i++)
				num_inputs += m_adj_matrix(i,j);

			SGVector<int32_t> input_indices(num_inputs);

			int32_t k = 0;
			for (int i=0; i<m_num_layers; i++)
			{
				if (m_adj_matrix(i,j))
				{
					input_indices[k] = i;
					k++;
				}
			}

			get_layer(j)->initialize(m_layers, input_indices);
		}
	}

	m_index_offsets = SGVector<int32_t>(m_num_layers);

	m_total_num_parameters = get_layer(0)->get_num_parameters();
	m_index_offsets[0] = 0;
	for (int32_t i=1; i<m_num_layers; i++)
	{
		m_index_offsets[i] = m_total_num_parameters;
		m_total_num_parameters += get_layer(i)->get_num_parameters();
	}

	m_params = SGVector<float64_t>(m_total_num_parameters);
	m_param_regularizable = SGVector<bool>(m_total_num_parameters);

	m_params.zero();
	m_param_regularizable.set_const(true);

	for (int32_t i=0; i<m_num_layers; i++)
	{
		SGVector<float64_t> layer_param = get_section(m_params, i);
		SGVector<bool> layer_param_regularizable =
			get_section(m_param_regularizable, i);

		get_layer(i)->initialize_parameters(layer_param,
			layer_param_regularizable, sigma);

		get_layer(i)->set_batch_size(m_batch_size);
	}
}

CNeuralNetwork::~CNeuralNetwork()
{
	SG_UNREF(m_layers);
}

CBinaryLabels* CNeuralNetwork::apply_binary(CFeatures* data)
{
	SGMatrix<float64_t> output_activations = forward_propagate(data);
	CBinaryLabels* labels = new CBinaryLabels(m_batch_size);

	for (int32_t i=0; i<m_batch_size; i++)
	{
		if (get_num_outputs()==1)
		{
			if (output_activations[i]>0.5) labels->set_label(i, 1);
			else labels->set_label(i, -1);
			
			labels->set_value(output_activations[i], i);
		}
		else if (get_num_outputs()==2)
		{
			float64_t v1 = output_activations[2*i];
			float64_t v2 = output_activations[2*i+1];
			if (v1>v2)
				labels->set_label(i, 1);
			else labels->set_label(i, -1);
			
			labels->set_value(v2/(v1+v2), i);
		}
	}
		
	return labels;
}

CRegressionLabels* CNeuralNetwork::apply_regression(CFeatures* data)
{
	SGMatrix<float64_t> output_activations = forward_propagate(data);
	SGVector<float64_t> labels_vec(m_batch_size);

	for (int32_t i=0; i<m_batch_size; i++)
			labels_vec[i] = output_activations[i];

	return new CRegressionLabels(labels_vec);
}


CMulticlassLabels* CNeuralNetwork::apply_multiclass(CFeatures* data)
{
	SGMatrix<float64_t> output_activations = forward_propagate(data);
	SGVector<float64_t> labels_vec(m_batch_size);

	for (int32_t i=0; i<m_batch_size; i++)
	{
		labels_vec[i] = CMath::arg_max(
			output_activations.matrix+i*get_num_outputs(), 1, get_num_outputs());
	}
	
	CMulticlassLabels* labels = new CMulticlassLabels(labels_vec);
	
	labels->allocate_confidences_for(get_num_outputs());
	for (int32_t i=0; i<m_batch_size; i++)
	{
		labels->set_multiclass_confidences(i, SGVector<float64_t>(
			output_activations.matrix, get_num_outputs(), i*get_num_outputs()));
	}
	
	return labels;
}

CDenseFeatures< float64_t >* CNeuralNetwork::transform(
	CDenseFeatures< float64_t >* data)
{
	SGMatrix<float64_t> output_activations = forward_propagate(data);
	return new CDenseFeatures<float64_t>(output_activations);
}

bool CNeuralNetwork::train_machine(CFeatures* data)
{
	REQUIRE(max_num_epochs>=0,
		"Maximum number of epochs (%i) must be >= 0\n", max_num_epochs);

	SGMatrix<float64_t> targets = labels_to_matrix(m_labels);

	for (int32_t i=0; i<m_num_layers-1; i++)
	{
		get_layer(i)->dropout_prop =
			get_layer(i)->is_input() ? dropout_input : dropout_hidden;
	}
	get_layer(m_num_layers-1)->dropout_prop = 0.0;

	m_is_training = true;
	for (int32_t i=0; i<m_num_layers; i++)
		get_layer(i)->is_training = true;

	bool result = 
		m_optimizer->optimize(this, dynamic_cast<CDotFeatures*>(data), targets);

	for (int32_t i=0; i<m_num_layers; i++)
		get_layer(i)->is_training = false;
	m_is_training = false;

	return result;
}

SGMatrix<float64_t> CNeuralNetwork::forward_propagate(CFeatures* data, int32_t j)
{
	set_batch_size(data->get_num_vectors());
	if (j==-1)
		j = m_num_layers-1;

	for (int32_t i=0; i<=j; i++)
	{
		CNeuralLayer* layer = get_layer(i);

		if (layer->is_input())
			layer->compute_activations(data);
		else
			layer->compute_activations(get_section(m_params, i), m_layers);

		layer->dropout_activations();
	}

	return get_layer(j)->get_activations();
}

float64_t CNeuralNetwork::compute_gradients(CDotFeatures* inputs,
		SGMatrix<float64_t> targets, SGVector<float64_t> gradients)
{
	forward_propagate(inputs);

	for (int32_t i=0; i<m_num_layers; i++)
		get_layer(i)->get_activation_gradients().zero();

	for (int32_t i=m_num_layers-1; i>=0; i--)
	{
		if (i==m_num_layers-1)
			get_layer(i)->compute_gradients(get_section(m_params,i), targets,
				m_layers, get_section(gradients,i));
		else
			get_layer(i)->compute_gradients(get_section(m_params,i),
				SGMatrix<float64_t>(), m_layers, get_section(gradients,i));
	}

	// L2 regularization
	if (l2_coefficient != 0.0)
	{
		for (int32_t i=0; i<m_total_num_parameters; i++)
		{
			if (m_param_regularizable[i])
				gradients[i] += l2_coefficient*m_params[i];
		}
	}

	// L1 regularization
	if (l1_coefficient != 0.0)
	{
		for (int32_t i=0; i<m_total_num_parameters; i++)
		{
			if (m_param_regularizable[i])
				gradients[i] +=
					l1_coefficient*CMath::sign<float64_t>(m_params[i]);
		}
	}
	
	// max-norm regularization
	if (max_norm != -1.0)
	{
		for (int32_t i=0; i<m_num_layers; i++)
		{
			SGVector<float64_t> layer_params = get_section(m_params,i);
			get_layer(i)->enforce_max_norm(layer_params, max_norm);
		}
	}

	return compute_error(targets);
}

float64_t CNeuralNetwork::compute_error(SGMatrix<float64_t> targets)
{
	float64_t error = get_layer(m_num_layers-1)->compute_error(targets);

	// L2 regularization
	if (l2_coefficient != 0.0)
	{
		for (int32_t i=0; i<m_total_num_parameters; i++)
		{
			if (m_param_regularizable[i])
				error += 0.5*l2_coefficient*m_params[i]*m_params[i];
		}
	}

	// L1 regularization
	if (l1_coefficient != 0.0)
	{
		for (int32_t i=0; i<m_total_num_parameters; i++)
		{
			if (m_param_regularizable[i])
				error += l1_coefficient*CMath::abs(m_params[i]);
		}
	}

	return error;
}

float64_t CNeuralNetwork::compute_error(CFeatures* inputs,
		SGMatrix<float64_t> targets)
{
	forward_propagate(inputs);
	return compute_error(targets);
}


float64_t CNeuralNetwork::check_gradients(float64_t approx_epsilon, float64_t s)
{
	// some random inputs and ouputs
	SGMatrix<float64_t> x(m_num_inputs,1);
	SGMatrix<float64_t> y(get_num_outputs(),1);

	for (int32_t i=0; i<x.num_rows; i++)
		x[i] = CMath::random(0.0,1.0);

	// the outputs are set up in the form of a probability distribution (in case
	// that is required by the output layer, i.e softmax)
	for (int32_t i=0; i<y.num_rows; i++)
		y[i] = CMath::random(0.0,1.0);

	float64_t y_sum = SGVector<float64_t>::sum(y.matrix, y.num_rows);
	for (int32_t i=0; i<y.num_rows; i++)
		y[i] /= y_sum;

	set_batch_size(1);

	// numerically compute gradients
	SGVector<float64_t> gradients_numerical(m_total_num_parameters);

	CDenseFeatures<float64_t>* xf = new CDenseFeatures<float64_t>(x);
	for (int32_t i=0; i<m_total_num_parameters; i++)
	{
		float64_t c =
			CMath::max<float64_t>(CMath::abs(approx_epsilon*m_params[i]),s);

		m_params[i] += c;
		float64_t error_plus = compute_error(xf,y);
		m_params[i] -= 2*c;
		float64_t error_minus = compute_error(xf,y);
		m_params[i] += c;

		gradients_numerical[i] = (error_plus-error_minus)/(2*c);
	}

	// compute gradients using backpropagation
	SGVector<float64_t> gradients_backprop(m_total_num_parameters);
	compute_gradients(xf, y, gradients_backprop);
	SG_UNREF(xf);

	float64_t sum = 0.0;
	for (int32_t i=0; i<m_total_num_parameters; i++)
	{
		sum += CMath::abs(gradients_backprop[i]-gradients_numerical[i]);
	}

	return sum/m_total_num_parameters;
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

SGMatrix<float64_t> CNeuralNetwork::features_to_matrix(CFeatures* features)
{
	REQUIRE(features != NULL, "Invalid (NULL) feature pointer\n");
	REQUIRE(features->get_feature_type() == F_DREAL,
		"Feature type must be F_DREAL\n");
	REQUIRE(features->get_feature_class() == C_DENSE,
		"Feature class must be C_DENSE\n");

	CDenseFeatures<float64_t>* inputs = (CDenseFeatures<float64_t>*) features;
	REQUIRE(inputs->get_num_features()==m_num_inputs,
		"Number of features (%i) must match the network's number of inputs "
		"(%i)\n", inputs->get_num_features(), get_num_inputs());

	SGMatrix<float64_t> matrix = inputs->get_feature_matrix();
	REQUIRE(matrix.matrix, "Feature matrix should not be empty");
	return matrix;
}

SGMatrix<float64_t> CNeuralNetwork::labels_to_matrix(CLabels* labs)
{
	REQUIRE(labs != NULL, "Invalid (NULL) labels pointer\n");

	SGMatrix<float64_t> targets(get_num_outputs(), labs->get_num_labels());
	targets.zero();

	if (labs->get_label_type() == LT_MULTICLASS)
	{
		CMulticlassLabels* labels_mc = (CMulticlassLabels*) labs;
		REQUIRE(labels_mc->get_num_classes()==get_num_outputs(),
			"Number of classes (%i) must match the network's number of "
			"outputs (%i)\n", labels_mc->get_num_classes(), get_num_outputs());

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
		REQUIRE(get_num_outputs() <= 2, "Cannot use %s in a neural network "
			"with more that 2 output neurons\n", lab->get_name());
	}
	else if (lab->get_label_type() == LT_REGRESSION)
	{
		REQUIRE(get_num_outputs() == 1, "Cannot use %s in a neural network "
			"with more that 1 output neuron\n", lab->get_name());
	}

	CMachine::set_labels(lab);
}

SGVector<float64_t>* CNeuralNetwork::get_layer_parameters(int32_t i)
{
	REQUIRE(i<m_num_layers && i >= 0, "Layer index (%i) out of range\n", i);

	int32_t n = get_layer(i)->get_num_parameters();
	SGVector<float64_t>* p = new SGVector<float64_t>(n);

	memcpy(p->vector, get_section(m_params, i), n*sizeof(float64_t));
	return p;
}

CNeuralLayer* CNeuralNetwork::get_layer(int32_t i)
{
	CNeuralLayer* layer = (CNeuralLayer*)m_layers->element(i);
	// needed because m_layers->element(i) increases the reference count of
	// layer i
	SG_UNREF(layer);
	return layer;
}

template <class T>
SGVector<T> CNeuralNetwork::get_section(SGVector<T> v, int32_t i)
{
	return SGVector<T>(v.vector+m_index_offsets[i],
		get_layer(i)->get_num_parameters(), false);
}

int32_t CNeuralNetwork::get_num_outputs()
{
	return get_layer(m_num_layers-1)->get_num_neurons();
}

CDynamicObjectArray* CNeuralNetwork::get_layers()
{
	SG_REF(m_layers);
	return m_layers;
}

void CNeuralNetwork::init()
{
	m_optimizer = new CLBFGSNeuralNetworkOptimizer();
	dropout_hidden = 0.0;
	dropout_input = 0.0;
	max_norm = -1.0;
	l2_coefficient = 0.0;
	l1_coefficient = 0.0;
	max_num_epochs = 0;
	epsilon = 1.0e-5;
	m_num_inputs = 0;
	m_num_layers = 0;
	m_layers = NULL;
	m_total_num_parameters = 0;
	m_batch_size = 1;
	m_is_training = false;

	SG_ADD((CSGObject**)&m_optimizer, "optimizer",
	       "Optimizer", MS_NOT_AVAILABLE);
	SG_ADD(&max_num_epochs, "max_num_epochs",
	       "Max number of Epochs", MS_NOT_AVAILABLE);
	SG_ADD(&epsilon, "epsilon",
	       "Epsilon", MS_NOT_AVAILABLE);
	SG_ADD(&m_num_inputs, "num_inputs",
	       "Number of Inputs", MS_NOT_AVAILABLE);
	SG_ADD(&m_num_layers, "num_layers",
	       "Number of Layers", MS_NOT_AVAILABLE);
	SG_ADD(&m_adj_matrix, "adj_matrix",
	       "Adjacency Matrix", MS_NOT_AVAILABLE);
	SG_ADD(&l2_coefficient, "l2_coefficient",
	       "L2 regularization coeff", MS_NOT_AVAILABLE);
	SG_ADD(&l1_coefficient, "l1_coefficient",
	       "L1 regularization coeff", MS_NOT_AVAILABLE);
	SG_ADD(&dropout_hidden, "dropout_hidden",
	       "Hidden neuron dropout probability", MS_NOT_AVAILABLE);
	SG_ADD(&dropout_input, "dropout_input",
	       "Input neuron dropout probability", MS_NOT_AVAILABLE);
	SG_ADD(&max_norm, "max_norm",
	       "Max Norm", MS_NOT_AVAILABLE);
	SG_ADD(&m_total_num_parameters, "total_num_parameters",
	       "Total number of parameters", MS_NOT_AVAILABLE);
	SG_ADD(&m_index_offsets, "index_offsets",
		"Index Offsets", MS_NOT_AVAILABLE);
	SG_ADD(&m_params, "params",
		"Parameters", MS_NOT_AVAILABLE);
	SG_ADD(&m_param_regularizable, "param_regularizable",
		"Parameter Regularizable", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_layers, "layers",
		"DynamicObjectArray of NeuralNetwork objects",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_is_training, "is_training",
		"is_training", MS_NOT_AVAILABLE);
}
