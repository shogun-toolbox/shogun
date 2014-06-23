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

CDeepBeliefNetwork::~CDeepBeliefNetwork()
{
	SG_UNREF(m_layer_sizes);
	SG_UNREF(m_layer_types);
	SG_UNREF(m_layer_input_offsets);
	SG_UNREF(m_rbms);
}

void CDeepBeliefNetwork::add_visible_layer(int32_t num_units, 
	ERBMVisibleUnitType unit_type, int32_t row_offset)
{
	m_layer_sizes->append_element(num_units);
	m_layer_types->append_element(unit_type);
	m_layer_input_offsets->append_element(row_offset);
	
	m_num_layers++;
	
	SGMatrix<bool> new_adj_matrix(m_num_layers,m_num_layers);
	new_adj_matrix.zero();
	
	for (int32_t i=0; i<m_num_layers-1; i++)
		for (int32_t j=0; j<m_num_layers-1; j++)
			new_adj_matrix(i,j) = m_adj_matrix(i,j);
	
	m_adj_matrix = new_adj_matrix;
}

void CDeepBeliefNetwork::add_hidden_layer(int32_t num_units)
{
	SGVector<int32_t> connection_indices(1);
	connection_indices[0] = m_num_layers-1;
	add_hidden_layer(num_units, connection_indices);
}

void CDeepBeliefNetwork::add_hidden_layer(int32_t num_units, 
	int32_t connection_index)
{
	SGVector<int32_t> connection_indices(1);
	connection_indices[0] = connection_index;
	add_hidden_layer(num_units, connection_indices);
}

void CDeepBeliefNetwork::add_hidden_layer(int32_t num_units, 
	int32_t connection1_index, int32_t connection2_index)
{
	SGVector<int32_t> connection_indices(2);
	connection_indices[0] = connection1_index;
	connection_indices[1] = connection2_index;
	add_hidden_layer(num_units, connection_indices);
}

void CDeepBeliefNetwork::add_hidden_layer(int32_t num_units, 
	SGVector< int32_t > connection_indices)
{
	m_layer_sizes->append_element(num_units);
	m_layer_types->append_element(-1);
	m_layer_input_offsets->append_element(0);
	
	m_num_layers++;
	
	SGMatrix<bool> new_adj_matrix(m_num_layers,m_num_layers);
	new_adj_matrix.zero();
	
	for (int32_t i=0; i<m_num_layers-1; i++)
		for (int32_t j=0; j<m_num_layers-1; j++)
			new_adj_matrix(i,j) = m_adj_matrix(i,j);
	
	m_adj_matrix = new_adj_matrix;
	
	for (int32_t i=0; i<connection_indices.vlen; i++)
		m_adj_matrix(m_num_layers-1, connection_indices[i]) = true;
}

void CDeepBeliefNetwork::initialize(float64_t sigma)
{
	// create one rbm for each hidden layer
	for (int32_t i=0; i<m_num_layers; i++)
	{
		if (m_layer_types->element(i)!=-1) continue;
		
		CRBM* rbm = new CRBM(m_layer_sizes->element(i));
		
		for (int32_t j=0; j<m_num_layers; j++)
		{
			if (m_adj_matrix(i,j))
			{
				int32_t group_type = m_layer_types->element(j);
				
				if (group_type==-1)
					group_type = RBMVUT_BINARY;
				
				rbm->add_visible_group(m_layer_sizes->element(j), 
					ERBMVisibleUnitType(group_type));
			}
		}
		
		rbm->initialize(sigma);
		
		m_rbms->append_element(rbm);
	}
	
	m_layer_rbm_indices = SGVector<int32_t>(m_num_layers);
	m_layer_rbm_group_indices = SGVector<int32_t>(m_num_layers);
	
	int32_t rbm_index = 0;
	for (int32_t i=0; i<m_num_layers; i++)
	{
		if (m_layer_types->element(i)==-1)
		{
			m_layer_rbm_group_indices[i] = 0;
			m_layer_rbm_indices[i] = rbm_index;
			rbm_index++;
		}
	}
	
	for (int32_t i=0; i<m_num_layers; i++)
	{
		if (m_layer_types->element(i)!=-1)
		{
			for (int32_t j=0; j<m_num_layers; j++)
			{
				if (m_adj_matrix(j,i))
				{
					m_layer_rbm_indices[i] = m_layer_rbm_indices[j];
					
					int32_t group_index = -1;
					for (int32_t k=0; k<=i; k++)
						if (m_adj_matrix(j,k))
							group_index++;
					
					m_layer_rbm_group_indices[i] = group_index;
					break;
				}
			}
		}
	}
}

void CDeepBeliefNetwork::pre_train(CDenseFeatures< float64_t >* features)
{
	for (int32_t k=0; k<m_rbms->get_num_elements(); k++)
	{	
		SG_INFO("Pre-training RBM %i\n",k);
		pre_train(k, features);
		SG_INFO("Finished pre-training RBM %i\n",k);
	}
}

void CDeepBeliefNetwork::pre_train(int32_t index, 
	CDenseFeatures< float64_t >* features)
{
	CRBM* rbm = get_rbm(index);
	
	for (int32_t i=0; i<index; i++)
	{
		CRBM* rbm_i = get_rbm(i); 
		rbm_i->set_batch_size(features->get_num_vectors());
		pack_into_visible_state_matrix(i, 
			features->get_feature_matrix(), rbm_i->visible_state);
		rbm_i->mean_hidden(rbm_i->visible_state, rbm_i->hidden_state);
	}
	
	SGMatrix<float64_t> rbm_feature_matrix(
		rbm->m_num_visible, features->get_num_vectors());
	pack_into_visible_state_matrix(index, 
		features->get_feature_matrix(), rbm_feature_matrix);
	CDenseFeatures<float64_t>* rbm_features = 
		new CDenseFeatures<float64_t>(rbm_feature_matrix);
		
	for (int32_t i=0; i<index; i++)
		get_rbm(i)->set_batch_size(1);
	
	rbm->train(rbm_features);
	
	SG_UNREF(rbm_features);
}

CDenseFeatures< float64_t >* CDeepBeliefNetwork::transform(
	CDenseFeatures< float64_t >* features)
{
	for (int32_t i=0; i<m_rbms->get_num_elements(); i++)
	{
		CRBM* rbm = get_rbm(i);
		rbm->set_batch_size(features->get_num_vectors());
		
		pack_into_visible_state_matrix(i, 
			features->get_feature_matrix(), rbm->visible_state);
		rbm->mean_hidden(rbm->visible_state, rbm->hidden_state);
	}
	
	return new CDenseFeatures<float64_t>(
		get_rbm(m_rbms->get_num_elements()-1)->hidden_state);
}

CNeuralNetwork* CDeepBeliefNetwork::convert_to_neural_network(
	CNeuralLayer* output_layer, float64_t sigma)
{
	CDynamicObjectArray* layers = new CDynamicObjectArray();
	
	for (int32_t i=0; i<m_num_layers; i++)
	{
		if (m_layer_types->element(i)!=-1)
			layers->append_element(new CNeuralInputLayer(
				m_layer_sizes->element(i), m_layer_input_offsets->element(i)));
		else
			layers->append_element(new CNeuralLogisticLayer(
				m_layer_sizes->element(i)));
	}
	if (output_layer!=NULL)
		layers->append_element(output_layer);
	
	CNeuralNetwork* network = new CNeuralNetwork(layers);
	
	for (int32_t i=0; i<m_num_layers; i++)
		for (int32_t j=0; j<m_num_layers; j++)
			if (m_adj_matrix(i,j))
				network->connect(j,i);
	
	if (output_layer!=NULL)
		network->connect(m_num_layers-1,m_num_layers);
	
	network->initialize(sigma);
	
	for (int32_t i=0; i<m_num_layers; i++)
	{
		if (m_layer_types->element(i)==-1)
		{
			CRBM* rbm = get_rbm(m_layer_rbm_indices[i]);
				
			int32_t N =  rbm->m_num_hidden + rbm->m_num_hidden*rbm->m_num_visible;
			
			float64_t* network_layer_params = 
				network->m_params.vector+network->m_index_offsets[i];
			
			for (int32_t j=0; j<N; j++)
				network_layer_params[j] = rbm->m_params[j+rbm->m_num_visible];
		}
	}
	
	return network;
}

// CDynamicObjectArray* CDeepBeliefNetwork::convert_to_neural_layers()
// {
// 	CDynamicObjectArray* layers = new CDynamicObjectArray();
// 	
// 	for (int32_t i=0; i<m_num_layers; i++)
// 	{
// 		if (m_layer_types->element(i)!=-1)
// 			layers->append_element(new CNeuralInputLayer(
// 				m_layer_sizes->element(i), m_layer_input_offsets->element(i)));
// 		else
// 			layers->append_element(new CNeuralLogisticLayer(
// 				m_layer_sizes->element(i)));
// 	}
// 	
// 	return layers;
// }
// 
// void CDeepBeliefNetwork::initialize_neural_network(CNeuralNetwork* network)
// {
// 	for (int32_t i=0; i<m_num_layers; i++)
// 	{
// 		if (m_layer_types->element(i)==-1)
// 		{
// 			CRBM* rbm = get_rbm(m_layer_rbm_indices[i]);
// 				
// 			int32_t N =  rbm->m_num_hidden + rbm->m_num_hidden*rbm->m_num_visible;
// 			
// 			float64_t* network_layer_params = 
// 				network->m_params.vector+network->m_index_offsets[i];
// 			
// 			for (int32_t j=0; j<N; j++)
// 				network_layer_params[j] = rbm->m_params[j+rbm->m_num_visible];
// 		}
// 	}
// }

void CDeepBeliefNetwork::sample(int32_t num_gibbs_steps, int32_t batch_size)
{
	for (int32_t i=0; i<m_rbms->get_num_elements(); i++)
		get_rbm(i)->set_batch_size(batch_size);
	
	get_rbm(m_rbms->get_num_elements()-1)->sample(num_gibbs_steps, batch_size);
	unpack_from_visible_state_matrix(m_rbms->get_num_elements()-1);
	
	for (int32_t i=m_rbms->get_num_elements()-1; i>=0; i--)
	{
		CRBM* rbm = get_rbm(i);
		rbm->mean_visible(rbm->hidden_state, rbm->visible_state);
		unpack_from_visible_state_matrix(i);
	}
}

CDenseFeatures< float64_t >* CDeepBeliefNetwork::sample_layer(int32_t V, 
	int32_t num_gibbs_steps, int32_t batch_size)
{
	sample(num_gibbs_steps, batch_size);
	
	return get_rbm(m_layer_rbm_indices[V])->sample_group(
		m_layer_rbm_group_indices[V], 0, batch_size);
}

void CDeepBeliefNetwork::sample_with_evidence(int32_t E, 
	CDenseFeatures< float64_t >* evidence, int32_t num_gibbs_steps)
{
	int32_t batch_size = evidence->get_num_vectors();
	
	for (int32_t i=0; i<m_rbms->get_num_elements(); i++)
		get_rbm(i)->set_batch_size(batch_size);
	
	int32_t n = m_rbms->get_num_elements()-1;
	get_rbm(n)->sample_with_evidence(
		m_layer_rbm_group_indices[E], evidence, num_gibbs_steps);
	unpack_from_visible_state_matrix(n);
	
	for (int32_t i=n; i>=0; i--)
	{
		CRBM* rbm = get_rbm(i);
		rbm->mean_visible(rbm->hidden_state, rbm->visible_state);
		unpack_from_visible_state_matrix(i);
	}
}

CDenseFeatures<float64_t>* CDeepBeliefNetwork::sample_layer_with_evidence(int32_t V, 
	int32_t E, CDenseFeatures< float64_t >* evidence, int32_t num_gibbs_steps)
{
	sample_with_evidence(E, evidence, num_gibbs_steps);
	
	return get_rbm(m_layer_rbm_indices[V])->sample_group(
		m_layer_rbm_group_indices[V], 0, evidence->get_num_vectors());
}

void CDeepBeliefNetwork::pack_into_visible_state_matrix(
	int32_t index, SGMatrix< float64_t > inputs, SGMatrix< float64_t > visible_state)
{
	int32_t hidden_layer_index = 0;
	for (int32_t i=0; i<m_num_layers; i++)
		if (m_layer_rbm_indices[i]==index && m_layer_types->element(i)==-1)
			hidden_layer_index = i;
	
	int32_t row_index = 0;
	for (int32_t k=0; k<m_num_layers; k++)
	{
		if (!m_adj_matrix(hidden_layer_index, k)) 
			continue;
		
		if (m_layer_types->element(k)==-1)
		{
			CRBM* input_rbm = get_rbm(m_layer_rbm_indices[k]);
			for (int32_t i=0; i<m_layer_sizes->element(k); i++)
			{
				for (int32_t j=0; j<inputs.num_cols; j++)
					visible_state(row_index, j) = input_rbm->hidden_state(i,j);
				
				row_index++;
			}
		}
		else
		{
			int32_t offset = m_layer_input_offsets->element(k);
			for (int32_t i=0; i<m_layer_sizes->element(k); i++)
			{
				for (int32_t j=0; j<inputs.num_cols; j++)
					visible_state(row_index, j) = inputs(i+offset,j);
				
				row_index++;
			}
		}
	}
}

void CDeepBeliefNetwork::unpack_from_visible_state_matrix(
	int32_t index)
{
	int32_t hidden_layer_index = 0;
	for (int32_t i=0; i<m_num_layers; i++)
		if (m_layer_rbm_indices[i]==index && m_layer_types->element(i)==-1)
			hidden_layer_index = i;
	
	SGMatrix<float64_t> visible_state = get_rbm(index)->visible_state;
	
	int32_t row_index = 0;
	for (int32_t k=0; k<m_num_layers; k++)
	{
		if (!m_adj_matrix(hidden_layer_index, k)) 
			continue;
		
		if (m_layer_types->element(k)==-1)
		{
			CRBM* input_rbm = get_rbm(m_layer_rbm_indices[k]);
			for (int32_t i=0; i<m_layer_sizes->element(k); i++)
			{
				for (int32_t j=0; j<visible_state.num_cols; j++)
					input_rbm->hidden_state(i,j) = visible_state(row_index, j);
				
				row_index++;
			}
		}
		else
			row_index += m_layer_sizes->element(k);
	}
}

void CDeepBeliefNetwork::init()
{
	m_num_layers = 0;
	m_layer_sizes = new CDynamicArray<int32_t>();
	m_layer_types = new CDynamicArray<int32_t>();
	m_layer_input_offsets = new CDynamicArray<int32_t>();
	m_rbms = new CDynamicObjectArray();
	
	SG_ADD(&m_num_layers, "num_layers","Number of layers", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_layer_sizes, "layer_sizes",
		"Size of each layer", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_layer_types, "m_layer_types",
		"Type of of each layer", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_layer_input_offsets, "layer_row_offsets",
		"Inputs row offset of of each layer", MS_NOT_AVAILABLE);
	SG_ADD(&m_layer_rbm_indices, "layer_rbm_indices",
		"RBM index of of each layer", MS_NOT_AVAILABLE);
	SG_ADD(&m_layer_rbm_group_indices, "layer_rbm_group_indices",
		"RBM group index of of each layer", MS_NOT_AVAILABLE);
	SG_ADD(&m_adj_matrix, "adj_matrix",
	    "Adjacency Matrix", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_rbms, "rbms", 
		"DynamicObjectArray of CRBM objects", MS_NOT_AVAILABLE);
}

CRBM* CDeepBeliefNetwork::get_rbm(int32_t index)
{
	CRBM* rbm = (CRBM*)m_rbms->element(index);
	SG_UNREF(rbm); // undo the SG_REF that CDynamicObjectArray::element() does
	return rbm;
}

#endif
