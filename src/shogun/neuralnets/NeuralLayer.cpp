/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 Khaled Nasr
 */

#include <shogun/base/Parameter.h>
#include <shogun/neuralnets/NeuralLayer.h>

using namespace shogun;

CNeuralLayer::CNeuralLayer() 
: CSGObject(), m_num_neurons(0), m_previous_layer_num_neurons(0),
	m_batch_size(0)
{
	init();
}


CNeuralLayer::CNeuralLayer(int32_t num_neurons)
: CSGObject(), m_num_neurons(num_neurons), 
m_previous_layer_num_neurons(0), m_batch_size(0)
{
	init();
}

CNeuralLayer::CNeuralLayer(const CNeuralLayer& orig) : CSGObject()
{
	shallow_copy(orig);
	init();
}


CNeuralLayer::~CNeuralLayer()
{
}

void CNeuralLayer::initialize(int32_t previous_layer_num_neurons)
{
	m_previous_layer_num_neurons = previous_layer_num_neurons;
}

void CNeuralLayer::set_batch_size(int32_t batch_size)
{
	m_batch_size = batch_size;
	
	if (m_activations.vector!=NULL) SG_FREE(m_activations.vector);
	if (m_input_gradients.vector!=NULL) SG_FREE(m_input_gradients.vector);
	if (m_local_gradients.vector!=NULL) SG_FREE(m_local_gradients.vector);
	
	m_activations.vlen = m_num_neurons * m_batch_size;
	m_input_gradients.vlen = m_previous_layer_num_neurons * m_batch_size;
	m_local_gradients.vlen = m_num_neurons * m_batch_size;
	
	m_activations.vector = SG_MALLOC(float64_t, m_activations.vlen);
	m_input_gradients.vector = SG_MALLOC(float64_t, m_input_gradients.vlen);
	m_local_gradients.vector = SG_MALLOC(float64_t, m_local_gradients.vlen);
}

void CNeuralLayer::init()
{
	SG_ADD(&m_num_neurons, "num_neurons",
	       "Number of Neurons", MS_NOT_AVAILABLE);
	SG_ADD(&m_previous_layer_num_neurons, "previous_layer_num_neurons",
	       "Number of neurons in the previous layer", MS_NOT_AVAILABLE);
	SG_ADD(&m_batch_size, "batch_size",
	       "Batch Size", MS_NOT_AVAILABLE);
	SG_ADD(&m_activations, "activations",
	       "Activations", MS_NOT_AVAILABLE);
	SG_ADD(&m_input_gradients, "input_gradients",
	       "Input Gradients", MS_NOT_AVAILABLE);
	SG_ADD(&m_local_gradients, "local_gradients",
	       "Local Gradients", MS_NOT_AVAILABLE);
}

void CNeuralLayer::shallow_copy(const CNeuralLayer &orig)
{
	m_num_neurons = orig.m_num_neurons;
	m_previous_layer_num_neurons = orig.m_previous_layer_num_neurons;
	m_batch_size = orig.m_batch_size;
	m_activations = SGVector<float64_t>(orig.m_activations);
	m_input_gradients = SGVector<float64_t>(orig.m_input_gradients);
	m_local_gradients = SGVector<float64_t>(orig.m_local_gradients);
}










