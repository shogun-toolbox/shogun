/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Elfarouk, Khaled Nasr
 */
#include <shogun/neuralnets/StanNeuralNetwork.h>
#include <shogun/mathematics/Math.h>
#include <shogun/optimization/lbfgs/lbfgs.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/lib/DynamicObjectArray.h>
#include <shogun/neuralnets/NeuralLayer.h>

 using namespace shogun;

 StanNeuralNetwork::StanNeuralNetwork()
 : CSGObject()
 {
 	init();
 }

 StanNeuralNetwork::StanNeuralNetwork(CDynamicObjectArray* layers)
 {
 	init();
 	set_layers(layers);
 }

 void StanNeuralNetwork::set_layers(CDynamicObjectArray* layers)
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

 void StanNeuralNetwork::connect(int32_t i, int32_t j)
 {
 	REQUIRE("i<j", "i(%i) must be less that j(%i)\n", i, j);
 	m_adj_matrix(i,j) = true;
 }

 void StanNeuralNetwork::quick_connect()
 {
 	m_adj_matrix.zero();
 	for (int32_t i=1; i<m_num_layers; i++)
 		m_adj_matrix(i-1, i) = true;
 }

 void StanNeuralNetwork::disconnect(int32_t i, int32_t j)
 {
 	m_adj_matrix(i,j) = false;
 }

 void StanNeuralNetwork::disconnect_all()
 {
 	m_adj_matrix.zero();
 }

 void StanNeuralNetwork::initialize_neural_network(float64_t sigma)
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

 			get_layer(j)->initialize_neural_layer(m_layers, input_indices);
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

 	m_params = StanVector(m_total_num_parameters, 1);

 	m_params.setZero(m_total_num_parameters,1);

 	for (int32_t i=0; i<m_num_layers; i++)
 	{
 		get_layer(i)->initialize_parameters(m_params, m_index_offsets[i],
      m_index_offsets[i] + get_layer(i)->get_num_parameters() -1 ,sigma);

 		get_layer(i)->set_batch_size(m_batch_size);
 	}
 }

 StanNeuralNetwork::~StanNeuralNetwork()
 {
 	SG_UNREF(m_layers);
 }

 SGMatrix<float64_t> StanNeuralNetwork::features_to_matrix(CFeatures* features)
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

 	return inputs->get_feature_matrix();
 }


 StanMatrix StanNeuralNetwork::forward_propagate(CFeatures* data, int32_t j)
 {
 	SGMatrix<float64_t> inputs = features_to_matrix(data);
 	set_batch_size(data->get_num_vectors());
 	return forward_propagate(inputs, j);
 }

 StanMatrix StanNeuralNetwork::forward_propagate(
 	SGMatrix<float64_t> inputs, int32_t j)
 {
 	if (j==-1)
 		j = m_num_layers-1;

 	for (int32_t i=0; i<=j; i++)
 	{
 		StanNeuralLayer* layer = get_layer(i);

 		if (layer->is_input())
      1==1; //compilation
 			//TODO layer->compute_activations(inputs);
 		else
 			layer->compute_activations(m_params, (int)(m_index_offsets[i]),
        (int)(m_index_offsets[i] + get_layer(i)->get_num_parameters() -1), m_layers);

 		layer->dropout_activations();
 	}
 	return get_layer(j)->get_activations();
 }

 StanNeuralLayer* StanNeuralNetwork::get_layer(int32_t i)
 {
 	auto layer = (StanNeuralLayer*)m_layers->element(i);
 	// needed because m_layers->element(i) increases the reference count of
 	// layer i
 	SG_UNREF(layer);
 	return layer;
 }

 int32_t StanNeuralNetwork::get_num_outputs()
 {
 	return get_layer(m_num_layers-1)->get_num_neurons();
 }

 CDynamicObjectArray* StanNeuralNetwork::get_layers()
 {
 	SG_REF(m_layers);
 	return m_layers;
 }

 void StanNeuralNetwork::init()
 {
 	m_num_inputs = 0;
 	m_num_layers = 0;
 	m_layers = NULL;
 	m_total_num_parameters = 0;
 	m_is_training = false;
  m_batch_size = 1;

 	SG_ADD(&m_num_inputs, "num_inputs",
 	       "Number of Inputs", MS_NOT_AVAILABLE);
 	SG_ADD(&m_num_layers, "num_layers",
 	       "Number of Layers", MS_NOT_AVAILABLE);
 	SG_ADD(&m_adj_matrix, "adj_matrix",
 	       "Adjacency Matrix", MS_NOT_AVAILABLE);
 	SG_ADD(&m_index_offsets, "index_offsets",
 		"Index Offsets", MS_NOT_AVAILABLE);
 	SG_ADD((CSGObject**)&m_layers, "layers",
 		"DynamicObjectArray of StanNeuralNetwork objects",
 		MS_NOT_AVAILABLE);
 	SG_ADD(&m_is_training, "is_training",
 		"is_training", MS_NOT_AVAILABLE);
 }
