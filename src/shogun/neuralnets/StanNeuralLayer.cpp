/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Elfarouk, Khaled Nasr
 */

#include <shogun/neuralnets/StanNeuralLayer.h>
#include <shogun/base/Parameter.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

StanNeuralLayer::StanNeuralLayer()
: CSGObject()
{
	init();
}


StanNeuralLayer::StanNeuralLayer(int32_t num_neurons)
: CSGObject()
{
	init();
	m_num_neurons = num_neurons;
	m_width = m_num_neurons;
	m_height = 1;
}

StanNeuralLayer::~StanNeuralLayer()
{
}

void StanNeuralLayer::initialize_neural_layer(CDynamicObjectArray* layers,
		SGVector< int32_t > input_indices)
{
	m_input_indices = input_indices;
	m_input_sizes = SGVector<int32_t>(input_indices.vlen);

	for (int32_t i=0; i<m_input_sizes.vlen; i++)
	{
		StanNeuralLayer* layer = (StanNeuralLayer*)layers->element(m_input_indices[i]);
		m_input_sizes[i] = layer->get_num_neurons();
		SG_UNREF(layer);
	}
}

void StanNeuralLayer::set_batch_size(int32_t batch_size)
{
	m_batch_size = batch_size;

  m_stan_activations.resize(m_num_neurons, m_batch_size);
	m_dropout_mask = SGMatrix<bool>(m_num_neurons, m_batch_size);
}

void StanNeuralLayer::dropout_activations()
{
	if (dropout_prop==0.0) return;

	if (is_training)
	{
    for(int32_t i=0; i<m_num_neurons; ++i)
    {
      for(int32_t j = 0; j<m_batch_size; ++j)
      {
        m_dropout_mask(i,j) = CMath::random(0.0,1.0) >= dropout_prop;
        m_stan_activations(i,j) *= m_dropout_mask(i,j);
      }
    }
	}
	else
	{
    for(int32_t i=0; i<m_num_neurons; ++i)
    {
      for(int32_t j = 0; j<m_batch_size; ++j)
      {
        m_stan_activations(i,j) *= (1.0 - dropout_prop);
      }
    }
	}
}

void StanNeuralLayer::init()
{
	m_num_neurons = 0;
	m_width = 0;
	m_height = 0;
	m_num_parameters = 0;
	m_batch_size = 0;
	dropout_prop = 0.0;
	is_training = false;
	autoencoder_position = NLAP_NONE;

	SG_ADD(&m_num_neurons, "num_neurons",
	       "Number of Neurons", MS_NOT_AVAILABLE);
	SG_ADD(&m_width, "width",
	       "Width", MS_NOT_AVAILABLE);
	SG_ADD(&m_height, "height",
	       "Height", MS_NOT_AVAILABLE);
	SG_ADD(&m_input_indices, "input_indices",
	       "Input Indices", MS_NOT_AVAILABLE);
	SG_ADD(&m_input_sizes, "input_sizes",
	       "Input Sizes", MS_NOT_AVAILABLE);
	SG_ADD(&dropout_prop, "dropout_prop",
	       "Dropout Probabilty", MS_NOT_AVAILABLE);
	SG_ADD(&is_training, "is_training",
	       "is_training", MS_NOT_AVAILABLE);
	SG_ADD(&m_batch_size, "batch_size",
	       "Batch Size", MS_NOT_AVAILABLE);
	SG_ADD(&m_dropout_mask, "dropout_mask",
	       "Dropout mask", MS_NOT_AVAILABLE);

	SG_ADD((machine_int_t*)&autoencoder_position, "autoencoder_position",
	       "Autoencoder Position", MS_NOT_AVAILABLE);
}
