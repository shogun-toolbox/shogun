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

 	m_params.zero();

 	for (int32_t i=0; i<m_num_layers; i++)
 	{
 		auto layer_param = get_section(m_params, i);

 		get_layer(i)->initialize_parameters(layer_param,sigma);

 		get_layer(i)->set_batch_size(m_batch_size);
 	}
 }

 StanNeuralNetwork::~StanNeuralNetwork()
 {
 	SG_UNREF(m_layers);
 }

//TODO: Fix apply_*
 CBinaryLabels* StanNeuralNetwork::apply_binary(CFeatures* data)
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

 CRegressionLabels* StanNeuralNetwork::apply_regression(CFeatures* data)
 {
 	SGMatrix<float64_t> output_activations = forward_propagate(data);
 	SGVector<float64_t> labels_vec(m_batch_size);

 	for (int32_t i=0; i<m_batch_size; i++)
 			labels_vec[i] = output_activations[i];

 	return new CRegressionLabels(labels_vec);
 }


 CMulticlassLabels* StanNeuralNetwork::apply_multiclass(CFeatures* data)
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

 SGMatrix<float64_t> StanNeuralNetwork::forward_propagate(CFeatures* data, int32_t j)
 {
 	SGMatrix<float64_t> inputs = features_to_matrix(data);
 	set_batch_size(data->get_num_vectors());
 	return forward_propagate(inputs, j);
 }

 SGMatrix<float64_t> StanNeuralNetwork::forward_propagate(
 	SGMatrix<float64_t> inputs, int32_t j)
 {
 	if (j==-1)
 		j = m_num_layers-1;

 	for (int32_t i=0; i<=j; i++)
 	{
 		CNeuralLayer* layer = get_layer(i);

 		if (layer->is_input())
 			layer->compute_activations(inputs);
 		else
 			layer->compute_activations(get_section(m_params, i), m_layers);

 		layer->dropout_activations();
 	}

 	return get_layer(j)->get_activations();
 }


 SGVector<float64_t>* StanNeuralNetwork::get_layer_parameters(int32_t i)
 {
 	REQUIRE(i<m_num_layers && i >= 0, "Layer index (%i) out of range\n", i);

 	int32_t n = get_layer(i)->get_num_parameters();
 	SGVector<float64_t>* p = new SGVector<float64_t>(n);

 	sg_memcpy(p->vector, get_section(m_params, i), n*sizeof(float64_t));
 	return p;
 }

 CNeuralLayer* StanNeuralNetwork::get_layer(int32_t i)
 {
 	CNeuralLayer* layer = (CNeuralLayer*)m_layers->element(i);
 	// needed because m_layers->element(i) increases the reference count of
 	// layer i
 	SG_UNREF(layer);
 	return layer;
 }

 template <class T>
 SGVector<T> StanNeuralNetwork::get_section(SGVector<T> v, int32_t i)
 {
 	return SGVector<T>(v.vector+m_index_offsets[i],
 		get_layer(i)->get_num_parameters(), false);
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

 	SG_ADD(&m_num_inputs, "num_inputs",
 	       "Number of Inputs", MS_NOT_AVAILABLE);
 	SG_ADD(&m_num_layers, "num_layers",
 	       "Number of Layers", MS_NOT_AVAILABLE);
 	SG_ADD(&m_adj_matrix, "adj_matrix",
 	       "Adjacency Matrix", MS_NOT_AVAILABLE);
 	SG_ADD(&m_index_offsets, "index_offsets",
 		"Index Offsets", MS_NOT_AVAILABLE);
 	SG_ADD(&m_params, "params",
 		"Parameters", MS_NOT_AVAILABLE);
 	SG_ADD((CSGObject**)&m_layers, "layers",
 		"DynamicObjectArray of StanNeuralNetwork objects",
 		MS_NOT_AVAILABLE);
 	SG_ADD(&m_is_training, "is_training",
 		"is_training", MS_NOT_AVAILABLE);
 }
