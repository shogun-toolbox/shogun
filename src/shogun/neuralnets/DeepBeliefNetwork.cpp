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

#include <shogun/neuralnets/DeepBeliefNetwork.h>

#ifdef HAVE_EIGEN3

#include <shogun/base/Parameter.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/lib/DynamicArray.h>
#include <shogun/neuralnets/NeuralNetwork.h>
#include <shogun/neuralnets/NeuralInputLayer.h>
#include <shogun/neuralnets/NeuralLogisticLayer.h>

using namespace shogun;

CDeepBeliefNetwork::CDeepBeliefNetwork() : CSGObject()
{
	init();
}

CDeepBeliefNetwork::CDeepBeliefNetwork(
	int32_t num_visible_units, ERBMVisibleUnitType unit_type) : CSGObject()
{
	init();
	m_layer_sizes->append_element(num_visible_units);
	m_num_layers++;
	m_visible_units_type = unit_type;
}

CDeepBeliefNetwork::~CDeepBeliefNetwork()
{
	SG_UNREF(m_layer_sizes);
}

void CDeepBeliefNetwork::add_hidden_layer(int32_t num_units)
{
	m_layer_sizes->append_element(num_units);
	m_num_layers++;
}

void CDeepBeliefNetwork::initialize(float64_t sigma)
{
	m_bias_index_offsets = SGVector<int32_t>(m_num_layers);
	m_weights_index_offsets = SGVector<int32_t>(m_num_layers-1);
	
	m_num_params = 0;
	for (int32_t i=0; i<m_num_layers; i++)
	{
		m_bias_index_offsets[i] = m_num_params;
		m_num_params += m_layer_sizes->element(i);
		
		if (i<m_num_layers-1)
		{
			m_weights_index_offsets[i] = m_num_params;
			m_num_params += m_layer_sizes->element(i+1)*m_layer_sizes->element(i);
		}
	}
	
	m_params = SGVector<float64_t>(m_num_params);
	for (int32_t i=0; i<m_num_params; i++)
		m_params[i] = CMath::normal_random(0.0,sigma);
	
	pt_cd_num_steps = SGVector<int32_t>(m_num_layers-1);
	pt_cd_num_steps.set_const(1);
	
	pt_cd_persistent = SGVector<bool>(m_num_layers-1);
	pt_cd_persistent.set_const(true);
	
	pt_cd_sample_visible = SGVector<bool>(m_num_layers-1);
	pt_cd_sample_visible.set_const(false);
	
	pt_l2_coefficient = SGVector<float64_t>(m_num_layers-1);
	pt_l2_coefficient.set_const(0.0);
	
	pt_l1_coefficient = SGVector<float64_t>(m_num_layers-1);
	pt_l1_coefficient.set_const(0.0);
	
	pt_monitoring_interval = SGVector<int32_t>(m_num_layers-1);
	pt_monitoring_interval.set_const(10);
	
	pt_monitoring_method = SGVector<int32_t>(m_num_layers-1);
	pt_monitoring_method.set_const(RBMMM_RECONSTRUCTION_ERROR);
	
	pt_max_num_epochs = SGVector<int32_t>(m_num_layers-1);
	pt_max_num_epochs.set_const(1);
	
	pt_gd_mini_batch_size = SGVector<int32_t>(m_num_layers-1);
	pt_gd_mini_batch_size.set_const(0);
	
	pt_gd_learning_rate = SGVector<float64_t>(m_num_layers-1);
	pt_gd_learning_rate.set_const(0.1);
	
	pt_gd_learning_rate_decay = SGVector<float64_t>(m_num_layers-1);
	pt_gd_learning_rate_decay.set_const(1.0);
	
	pt_gd_momentum = SGVector<float64_t>(m_num_layers-1);
	pt_gd_momentum.set_const(0.9);
}

void CDeepBeliefNetwork::set_batch_size(int32_t batch_size)
{
	if (m_batch_size == batch_size) return;
	
	m_batch_size = batch_size;
	
	m_states = SGMatrixList<float64_t>(m_num_layers);
	
	for (int32_t i=0; i<m_num_layers; i++)
		m_states.set_matrix(i, SGMatrix<float64_t>(m_layer_sizes->element(i), m_batch_size));
	
	reset_chain();
}

void CDeepBeliefNetwork::pre_train(CDenseFeatures< float64_t >* features)
{
	for (int32_t k=0; k<m_num_layers-1; k++)
	{	
		SG_INFO("Pre-training RBM %i\n",k);
		pre_train(k, features);
		SG_INFO("Finished pre-training RBM %i\n",k);
	}
}

void CDeepBeliefNetwork::pre_train(int32_t index, 
	CDenseFeatures< float64_t >* features)
{
	CRBM rbm(m_layer_sizes->element(index+1));
	if (index == 0)
		rbm.add_visible_group(m_layer_sizes->element(index), m_visible_units_type);
	else
		rbm.add_visible_group(m_layer_sizes->element(index), RBMVUT_BINARY);
	rbm.initialize(m_sigma);
	
	rbm.cd_num_steps = pt_cd_num_steps[index];
	rbm.cd_persistent = pt_cd_persistent[index];
	rbm.cd_sample_visible = pt_cd_sample_visible[index];
	rbm.l2_coefficient = pt_l2_coefficient[index];
	rbm.l1_coefficient = pt_l1_coefficient[index];
	rbm.monitoring_interval = pt_monitoring_interval[index];
	rbm.monitoring_method = ERBMMonitoringMethod(pt_monitoring_method[index]);
	rbm.max_num_epochs = pt_max_num_epochs[index];
	rbm.gd_mini_batch_size = pt_gd_mini_batch_size[index];
	rbm.gd_learning_rate = pt_gd_learning_rate[index];
	rbm.gd_learning_rate_decay = pt_gd_learning_rate_decay[index];
	rbm.gd_momentum = pt_gd_momentum[index];

	if (index > 0)
	{
		CDenseFeatures<float64_t>* transformed_features = 
			transform(features, index);
		rbm.train(transformed_features);
		SG_UNREF(transformed_features);
	}
	else
		rbm.train(features);
	
	SGVector<float64_t> rbm_b = rbm.get_visible_bias();
	SGVector<float64_t> rbm_c = rbm.get_hidden_bias();
	SGMatrix<float64_t> rbm_w = rbm.get_weights();
	
	SGVector<float64_t> dbn_b = get_biases(index);
	SGVector<float64_t> dbn_c = get_biases(index+1);
	SGMatrix<float64_t> dbn_w = get_weights(index);
	
	for (int32_t i=0; i<dbn_b.vlen; i++)
		dbn_b[i] = rbm_b[i];
	
	for (int32_t i=0; i<dbn_c.vlen; i++)
		dbn_c[i] = rbm_c[i];
	
	for (int32_t i=0; i<dbn_w.num_rows*dbn_w.num_cols; i++)
		dbn_w[i] = rbm_w[i];
}

void CDeepBeliefNetwork::train(CDenseFeatures<float64_t>* features)
{
	REQUIRE(features != NULL, "Invalid (NULL) feature pointer\n");
	REQUIRE(features->get_num_features()==m_layer_sizes->element(0), 
		"Number of features (%i) must match the DBN's number of visible units "
		"(%i)\n", features->get_num_features(), m_layer_sizes->element(0));
	
	SGMatrix<float64_t> inputs = features->get_feature_matrix();
	
	int32_t training_set_size = inputs.num_cols;
	if (gd_mini_batch_size==0) gd_mini_batch_size = training_set_size;
	set_batch_size(gd_mini_batch_size);
	
	SGVector<float64_t> rec_params(m_num_params);
	for (int32_t i=0; i<rec_params.vlen; i++)
		rec_params[i] = m_params[i];
		
	SGVector<float64_t> gradients(m_num_params);
	SGVector<float64_t> rec_gradients(m_num_params);
	gradients.zero();
	rec_gradients.zero();
	
	SGVector<float64_t> param_updates(m_num_params);
	SGVector<float64_t> rec_param_updates(m_num_params);
	param_updates.zero();
	rec_param_updates.zero();
	
	SGMatrixList<float64_t> sleep_states = m_states;
	SGMatrixList<float64_t> wake_states(m_num_layers);
	SGMatrixList<float64_t> psleep_states(m_num_layers);
	SGMatrixList<float64_t> pwake_states(m_num_layers);
	
	for (int32_t i=0; i<m_num_layers; i++)
	{
		wake_states.set_matrix(i, SGMatrix<float64_t>(m_layer_sizes->element(i), m_batch_size));
		psleep_states.set_matrix(i, SGMatrix<float64_t>(m_layer_sizes->element(i), m_batch_size));
		pwake_states.set_matrix(i, SGMatrix<float64_t>(m_layer_sizes->element(i), m_batch_size));
	}
	
	CRBM top_rbm(m_layer_sizes->element(m_num_layers-1));
	if (m_num_layers > 2)
		top_rbm.add_visible_group(m_layer_sizes->element(m_num_layers-2), RBMVUT_BINARY);
	else
		top_rbm.add_visible_group(m_layer_sizes->element(0), m_visible_units_type);
	
	top_rbm.initialize();
	top_rbm.m_params = SGVector<float64_t>(
		m_params.vector+m_bias_index_offsets[m_num_layers-2], 
		top_rbm.get_num_parameters(), false);

	top_rbm.cd_num_steps = cd_num_steps;
	top_rbm.cd_persistent = false;
	top_rbm.set_batch_size(gd_mini_batch_size);
	
	float64_t alpha = gd_learning_rate;
	
	int32_t counter = 0;
	for (int32_t i=0; i<max_num_epochs; i++)
	{	
		for (int32_t j=0; j < training_set_size; j += gd_mini_batch_size)
		{
			alpha = gd_learning_rate_decay*alpha;
			
			if (j+gd_mini_batch_size>training_set_size) 
				j = training_set_size-gd_mini_batch_size;

			SGMatrix<float64_t> inputs_batch(inputs.matrix+j*inputs.num_rows, 
				inputs.num_rows, gd_mini_batch_size, false);
			
			for (int32_t k=0; k<m_num_params; k++)
			{
				m_params[k] += gd_momentum*param_updates[k];
				rec_params[k] += gd_momentum*rec_param_updates[k];
			}
			
			wake_sleep(inputs_batch, &top_rbm, sleep_states, wake_states, 
				psleep_states, pwake_states, m_params, 
				rec_params, gradients, rec_gradients);
			
			for (int32_t k=0; k<m_num_params; k++)
			{
				param_updates[k] = gd_momentum*param_updates[k]
						-alpha*gradients[k];
				m_params[k] -= alpha*gradients[k];
				
				rec_param_updates[k] = gd_momentum*rec_param_updates[k]
						-alpha*rec_gradients[k];
				rec_params[k] -= alpha*rec_gradients[k];
			}
			
			if (counter%monitoring_interval == 0)
			{
				SGMatrix<float64_t> reconstruction = sleep_states[0];
				float64_t error = 0;
				for (int32_t i=0; i<inputs_batch.num_rows*inputs_batch.num_cols; i++)
					error += CMath::pow(reconstruction[i]-inputs_batch[i],2);
		
				error /= m_batch_size;
	
				SG_INFO("Epoch %i: reconstruction Error = %f\n",i, error);
			}
			counter++;
		}
	}
}

CDenseFeatures< float64_t >* CDeepBeliefNetwork::transform(
	CDenseFeatures< float64_t >* features, int32_t i)
{
	if (i==-1)
		i = m_num_layers-1;
	
	SGMatrix<float64_t> transformed_feature_matrix = features->get_feature_matrix();
	for (int32_t h=1; h<=i; h++)
	{
		SGMatrix<float64_t> m(m_layer_sizes->element(h), features->get_num_vectors());
		up_step(h, m_params, transformed_feature_matrix, m, false);
		transformed_feature_matrix = m;
	}
	
	return new CDenseFeatures<float64_t>(transformed_feature_matrix);
}

CDenseFeatures<float64_t>* CDeepBeliefNetwork::sample(
	int32_t num_gibbs_steps, int32_t batch_size)
{
	set_batch_size(batch_size);
	
	for (int32_t i=0; i<num_gibbs_steps; i++)
	{
		up_step(m_num_layers-1, m_params, m_states[m_num_layers-2], 
			m_states[m_num_layers-1]);
		down_step(m_num_layers-2, m_params, m_states[m_num_layers-1], 
			m_states[m_num_layers-2]);
	}
	
	for (int32_t i=m_num_layers-3; i>=0; i--)
		down_step(i, m_params, m_states[i+1], m_states[i]);
	
	return new CDenseFeatures<float64_t>(m_states[0]);
}

void CDeepBeliefNetwork::reset_chain()
{
	SGMatrix<float64_t> s = m_states[m_num_layers-2];
	
	for (int32_t i=0; i<s.num_rows*s.num_cols; i++)
		s[i] = CMath::random(0.0,1.0) > 0.5;
}

CNeuralNetwork* CDeepBeliefNetwork::convert_to_neural_network(
	CNeuralLayer* output_layer, float64_t sigma)
{
	CDynamicObjectArray* layers = new CDynamicObjectArray();
	
	layers->append_element(new CNeuralInputLayer(m_layer_sizes->element(0)));
	
	for (int32_t i=1; i<m_num_layers; i++)
		layers->append_element(new CNeuralLogisticLayer(m_layer_sizes->element(i)));
	
	if (output_layer!=NULL)
		layers->append_element(output_layer);
	
	CNeuralNetwork* network = new CNeuralNetwork(layers);
	
	network->quick_connect();
	network->initialize(sigma);

	for (int32_t i=1; i<m_num_layers; i++)
	{
		SGMatrix<float64_t> W = get_weights(i-1);
		SGVector<float64_t> b = get_biases(i);
		
		for (int32_t j=0; j<b.vlen; j++)
			network->m_params[j+network->m_index_offsets[i]] = b[j]; 
		
		for (int32_t j=0; j<W.num_rows*W.num_cols; j++)
			network->m_params[j+network->m_index_offsets[i]+b.vlen] = W[j]; 
	}
	
	return network;
}

void CDeepBeliefNetwork::down_step(int32_t index, SGVector< float64_t > params, 
	SGMatrix< float64_t > input, SGMatrix< float64_t > result, bool sample_states)
{
	typedef Eigen::Map<Eigen::MatrixXd> EMatrix;
	typedef Eigen::Map<Eigen::VectorXd> EVector;
	
	EMatrix In(input.matrix, input.num_rows, input.num_cols);
	EMatrix Out(result.matrix, result.num_rows, result.num_cols);
	EVector B(get_biases(index,params).vector, m_layer_sizes->element(index));
	
	Out.colwise() = B;
	
	if (index < m_num_layers-1);
	{
		EMatrix W(get_weights(index,params).matrix, 
			m_layer_sizes->element(index+1), m_layer_sizes->element(index));
		Out += W.transpose()*In;
	}
	
	if (index > 0 || (index==0 && m_visible_units_type==RBMVUT_BINARY))
	{
		int32_t len = m_layer_sizes->element(index)*m_batch_size;
		for (int32_t i=0; i<len; i++)
				result[i] = 1.0/(1.0+CMath::exp(-1.0*result[i]));
	}
	
	if (index == 0 && m_visible_units_type==RBMVUT_SOFTMAX)
	{
		float64_t max = Out.maxCoeff();
		
		for (int32_t j=0; j<m_batch_size; j++)
		{
			float64_t sum = 0;
			for (int32_t i=0; i<m_layer_sizes->element(0); i++)
				sum += CMath::exp(Out(i,j)-max);
			
			float64_t normalizer = CMath::log(sum);
			for (int32_t k=0; k<m_layer_sizes->element(0); k++)
				Out(k,j) = CMath::exp(Out(k,j)-max-normalizer);
		}
	}
	
	if (sample_states && index>0)
	{
		int32_t len = m_layer_sizes->element(index)*m_batch_size;
		for (int32_t i=0; i<len; i++)
			result[i] = CMath::random(0.0,1.0) < result[i];
	}
}

void CDeepBeliefNetwork::up_step(int32_t index, SGVector< float64_t > params, 
	SGMatrix< float64_t > input, SGMatrix< float64_t > result, bool sample_states)
{
	typedef Eigen::Map<Eigen::MatrixXd> EMatrix;
	typedef Eigen::Map<Eigen::VectorXd> EVector;
	
	EMatrix In(input.matrix, input.num_rows, input.num_cols);
	EMatrix Out(result.matrix, result.num_rows, result.num_cols);
	EVector C(get_biases(index, params).vector, m_layer_sizes->element(index));
	
	Out.colwise() = C;
	
	if (index>0)
	{
		EMatrix W(get_weights(index-1, params).matrix, 
			m_layer_sizes->element(index), m_layer_sizes->element(index-1));
		Out += W*In;
	}
	
	int32_t len = result.num_rows*result.num_cols;
	for (int32_t i=0; i<len; i++)
		result[i] = 1.0/(1.0+CMath::exp(-1.0*result[i]));
	
	if (sample_states && index>0)
	{
		for (int32_t i=0; i<len; i++)
			result[i] = CMath::random(0.0,1.0) < result[i];
	}
}

void CDeepBeliefNetwork::wake_sleep(SGMatrix< float64_t > data, CRBM* top_rbm, 
	SGMatrixList<float64_t> sleep_states, SGMatrixList<float64_t> wake_states, 
	SGMatrixList<float64_t> psleep_states, SGMatrixList<float64_t> pwake_states,
	SGVector<float64_t> gen_params,
	SGVector<float64_t> rec_params,
	SGVector<float64_t> gen_gradients,
	SGVector<float64_t> rec_gradients)
{
	typedef Eigen::Map<Eigen::MatrixXd> EMatrix;
	typedef Eigen::Map<Eigen::VectorXd> EVector;
	
	// Wake phase
	for (int32_t i=0; i<data.num_rows*data.num_cols; i++)
		wake_states[0][i] = data[i];
	
	for (int32_t i=1; i<m_num_layers-1; i++)
		up_step(i, rec_params, wake_states[i-1], wake_states[i]);
	
	// Contrastive divergence in the top RBM
	SGVector<float64_t> top_rbm_gradients(
		gen_gradients.vector+m_bias_index_offsets[m_num_layers-2],
		top_rbm->get_num_parameters(), false);
	top_rbm->contrastive_divergence(wake_states[m_num_layers-2], top_rbm_gradients);
	
	// Sleep phase
	sleep_states.set_matrix(m_num_layers-2, top_rbm->visible_state);
	for (int32_t i=m_num_layers-3; i>=0; i--)
		down_step(i, gen_params, sleep_states[i+1], sleep_states[i]);
	
	// Predictions
	for (int32_t i=1; i<m_num_layers-1; i++)
		up_step(i, rec_params, sleep_states[i-1], psleep_states[i]);
	for (int32_t i=0; i<m_num_layers-2; i++)
		down_step(i, gen_params, wake_states[i+1], pwake_states[i]);
	
	// Gradients for generative parameters
	for (int32_t i=0; i<m_num_layers-2; i++)
	{
		EMatrix wake_i(wake_states[i].matrix, 
			wake_states[i].num_rows, wake_states[i].num_cols);
		EMatrix wake_i_plus_one(wake_states[i+1].matrix, 
			wake_states[i+1].num_rows, wake_states[i+1].num_cols);
		EMatrix pwake_i(pwake_states[i].matrix, 
			pwake_states[i].num_rows, pwake_states[i].num_cols);
		
		EMatrix WG_gen(get_weights(i,gen_gradients).matrix, 
			m_layer_sizes->element(i+1), m_layer_sizes->element(i));
		EVector BG_gen(get_biases(i,gen_gradients).vector, m_layer_sizes->element(i));
		
		pwake_i = pwake_i - wake_i;
		BG_gen = pwake_i.rowwise().sum()/m_batch_size;
		WG_gen = wake_i_plus_one*pwake_i.transpose()/m_batch_size;
	}
	
	// Gradients for reconstruction parameters
	for (int32_t i=1; i<m_num_layers-1; i++)
	{
		EMatrix sleep_i(sleep_states[i].matrix, 
			sleep_states[i].num_rows, sleep_states[i].num_cols);
		EMatrix psleep_i(psleep_states[i].matrix, 
			psleep_states[i].num_rows, psleep_states[i].num_cols);
		EMatrix sleep_i_minus_one(sleep_states[i-1].matrix, 
			sleep_states[i-1].num_rows, sleep_states[i-1].num_cols);
		
		EMatrix WG_rec(get_weights(i-1,rec_gradients).matrix, 
			m_layer_sizes->element(i), m_layer_sizes->element(i-1));
		EVector BG_rec(get_biases(i,rec_gradients).vector, m_layer_sizes->element(i));
		
		psleep_i = psleep_i - sleep_i;
		BG_rec = psleep_i.rowwise().sum()/m_batch_size;
		WG_rec = psleep_i*sleep_i_minus_one.transpose()/m_batch_size;
	}
}

SGMatrix< float64_t > CDeepBeliefNetwork::get_weights(int32_t i, 
	SGVector< float64_t > p)
{
	if (p.vlen==0)
		return SGMatrix<float64_t>(m_params.vector+m_weights_index_offsets[i], 
			m_layer_sizes->element(i+1), m_layer_sizes->element(i), false);
	else
		return SGMatrix<float64_t>(p.vector+m_weights_index_offsets[i], 
			m_layer_sizes->element(i+1), m_layer_sizes->element(i), false);
}

SGVector< float64_t > CDeepBeliefNetwork::get_biases(int32_t i, 
	SGVector< float64_t > p)
{
	if (p.vlen==0)
		return SGVector<float64_t>(m_params.vector+m_bias_index_offsets[i], 
			m_layer_sizes->element(i), false);
	else
		return SGVector<float64_t>(p.vector+m_bias_index_offsets[i], 
			m_layer_sizes->element(i), false);;
}

void CDeepBeliefNetwork::init()
{
	cd_num_steps = 1;
	monitoring_interval = 10;
	
	gd_mini_batch_size = 0;
	max_num_epochs = 1; 
	gd_learning_rate = 0.1; 
	gd_learning_rate_decay = 1.0;
	gd_momentum = 0.9; 
	
	m_visible_units_type = RBMVUT_BINARY;
	m_num_layers = 0;
	m_layer_sizes = new CDynamicArray<int32_t>();
	m_batch_size = 0;
	m_num_params = 0;
	m_sigma = 0.01;
	
	SG_ADD((machine_int_t*)&m_visible_units_type, "visible_units_type", 
		"Type of the visible units", MS_NOT_AVAILABLE);
	SG_ADD(&m_num_layers, "num_layers",
		"Number of layers", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_layer_sizes, "layer_sizes",
		"Size of each hidden layer", MS_NOT_AVAILABLE);
	
	SG_ADD(&m_params, "params",
		"Parameters of the network", MS_NOT_AVAILABLE);
	SG_ADD(&m_num_params, "num_params",
		"Number of parameters", MS_NOT_AVAILABLE);
	SG_ADD(&m_bias_index_offsets, "bias_index_offsets",
		"Index offsets of the biases", MS_NOT_AVAILABLE);
	SG_ADD(&m_weights_index_offsets, "weights_index_offsets",
		"Index offsets of the weights", MS_NOT_AVAILABLE);
	
	SG_ADD(&pt_cd_num_steps, "pt_cd_num_steps",
	    "Pre-training Number of CD Steps", MS_NOT_AVAILABLE);
	SG_ADD(&pt_cd_persistent, "pt_cd_persistent",
	    "Pre-training Persistent CD", MS_NOT_AVAILABLE);
	SG_ADD(&pt_cd_sample_visible, "pt_cd_sample_visible",
	    "Pre-training Number of CD Sample Visible", MS_NOT_AVAILABLE);
	SG_ADD(&pt_l2_coefficient, "pt_l2_coefficient",
	    "Pre-training L2 regularization coeff", MS_NOT_AVAILABLE);
	SG_ADD(&pt_l1_coefficient, "pt_l1_coefficient",
	    "Pre-training L1 regularization coeff", MS_NOT_AVAILABLE);
	SG_ADD(&pt_monitoring_interval, "pt_monitoring_interval",
	    "Pre-training Monitoring Interval", MS_NOT_AVAILABLE);
	SG_ADD(&pt_monitoring_method, "pt_monitoring_method",
	    "Pre-training Monitoring Method", MS_NOT_AVAILABLE);
	SG_ADD(&pt_cd_num_steps, "pt_gd_mini_batch_size",
	    "Pre-training Gradient Descent Mini-batch size", MS_NOT_AVAILABLE);
	SG_ADD(&pt_max_num_epochs, "pt_max_num_epochs",
	    "Pre-training Max number of Epochs", MS_NOT_AVAILABLE);
	SG_ADD(&pt_gd_learning_rate, "pt_gd_learning_rate",
	    "Pre-training Gradient descent learning rate", MS_NOT_AVAILABLE);
	SG_ADD(&pt_gd_learning_rate_decay, "pt_gd_learning_rate_decay",
	    "Pre-training Gradient descent learning rate decay", MS_NOT_AVAILABLE);
	SG_ADD(&pt_gd_momentum, "pt_gd_momentum",
	    "Pre-training Gradient Descent Momentum", MS_NOT_AVAILABLE);
	
	SG_ADD(&cd_num_steps, "cd_num_steps", "Number of CD Steps", MS_NOT_AVAILABLE);
	SG_ADD(&monitoring_interval, "monitoring_interval", 
		"Monitoring Interval", MS_NOT_AVAILABLE);
	
	SG_ADD(&gd_mini_batch_size, "gd_mini_batch_size",
	       "Gradient Descent Mini-batch size", MS_NOT_AVAILABLE);
	SG_ADD(&max_num_epochs, "max_num_epochs",
	       "Max number of Epochs", MS_NOT_AVAILABLE);
	SG_ADD(&gd_learning_rate, "gd_learning_rate",
	       "Gradient descent learning rate", MS_NOT_AVAILABLE);
	SG_ADD(&gd_learning_rate_decay, "gd_learning_rate_decay",
	       "Gradient descent learning rate decay", MS_NOT_AVAILABLE);
	SG_ADD(&gd_momentum, "gd_momentum",
	       "Gradient Descent Momentum", MS_NOT_AVAILABLE);
	
	SG_ADD(&m_sigma, "m_sigma", "Initialization Sigma", MS_NOT_AVAILABLE);
}

#endif
